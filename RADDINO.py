"""
RAD-DINO -- CXR 4-class classification (normal / bacteria / virus / COVID-19)

RADDINO.py, ConvNeXtTiny.py and ResNet.py share the same code; only the MODEL-SPECIFIC block differs.

Modes
-----
train : train, pick the best epoch by validation macro-F1, save best_weight.pth, curves.png,
        confusion_matrix_best_val.png and training_info.txt, then predict the test set with the best weight
        (output_csv, test_probs.csv, testing_info.txt).
        --cv_folds K (K >= 2): train+val are merged and split into K stratified folds; every fold is trained in
        <run>/fold<k>/, the run folder gets the out-of-fold (OOF) summary and the K-fold test ensemble.
        COVID-19 threshold (--covid_thr auto): t is chosen on the OOF probabilities, the OOF score with the
        threshold is reported by nested CV, and <output_csv>_thr.csv is written next to the argmax output_csv.
test  : load --weight (one .pth, several .pth, or a run folder -> all best_weight.pth below it, softmax averaged)
        and predict --test_csv. If the csv has labels (e.g. val_data.csv), metrics + confusion matrix are reported.
        With a CV run folder as --weight, --covid_thr auto takes t from its fold<k>/val_probs.csv (no retraining).

Examples (run from any folder)
--------
python RADDINO.py --mode train --run_name base
python RADDINO.py --mode train --run_name base_cv5 --cv_folds 5
python RADDINO.py --mode train --run_name rot10 --rotation 10
python RADDINO.py --mode test --weight Output/RADDINO/base/best_weight.pth
python RADDINO.py --mode test --weight Output/RADDINO/base_cv5 --run_name base_cv5_test
python RADDINO.py --mode test --weight Output/RADDINO/base/best_weight.pth --test_csv ../cxr-dataset/val_data.csv
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # noqa: F401  (models: used by the CNN scripts)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["normal", "bacteria", "virus", "COVID-19"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixed training settings shared by all three scripts (the defaults of the previous experiments).
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
WARMUP_EPOCHS = 1      # linear warmup, then cosine decay to 0 (stepped every batch)
SGD_MOMENTUM = 0.9     # --optimizer sgd: Nesterov momentum
USE_AMP = True         # mixed precision on CUDA
# COVID-19 threshold grid (same as the previous experiments, cv_summary.py): 0 = plain argmax, then 0.30 ... 0.99
THR_GRID = np.concatenate([[0.0], np.round(np.arange(0.30, 0.991, 0.01), 2)])
COVID = 3              # index of COVID-19 in CLASSES


# ============================== MODEL-SPECIFIC ============================== #
MODEL_NAME = "RADDINO"
MODEL_DESC = "RAD-DINO (microsoft/rad-dino): DINOv2 ViT-B/14, self-supervised on chest X-rays"
# argument defaults of this model (settings of the previous RAD-DINO experiments, stage 3-6)
DEFAULTS = {"img_size": 392, "epochs": 20, "lr": 1e-4, "weight_decay": 0.05, "batch_size": 16, "patience": 0}
HF_REPO = "microsoft/rad-dino"
DROPOUT = 0.2          # dropout before the final linear layer
DROP_PATH = 0.1        # stochastic depth rate
LLRD = 0.75            # layer-wise lr decay: block i (0..L-1) gets lr * LLRD^(L-i), embeddings LLRD^(L+1), head lr
WD_SKIP_NORM = True    # no weight decay on 1-D parameters (norm weights, biases, layer scales) and cls / mask tokens
MEAN, STD = [0.5307] * 3, [0.2583] * 3   # RAD-DINO preprocessor_config.json
PATCH = 14
FIXED_INFO = {"hf_repo": HF_REPO, "head": "Dropout(0.2) + Linear on [CLS, mean(patch tokens)] (1536-d)",
              "dropout": DROPOUT, "drop_path": DROP_PATH, "llrd": LLRD, "wd_skip_norm": WD_SKIP_NORM,
              "norm_mean": MEAN, "norm_std": STD}


class RadDinoBackbone(nn.Module):
    """Hugging Face Dinov2Model -> [CLS token, mean of the patch tokens] of the final-layer-normed tokens.
    The position embeddings are interpolated by the model for any input size (use a multiple of 14)."""

    def __init__(self, pretrained):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(HF_REPO)
        cfg.drop_path_rate = DROP_PATH
        self.m = AutoModel.from_pretrained(HF_REPO, config=cfg) if pretrained else AutoModel.from_config(cfg)
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.out_dim = cfg.hidden_size * 2

    def forward(self, x):
        t = self.m(pixel_values=x).last_hidden_state
        return torch.cat([t[:, 0], t[:, self.n_prefix:].mean(1)], 1)


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(backbone.out_dim, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(pretrained=True):
    """-> model whose state_dict keys match the old main.py checkpoints (hf:microsoft/rad-dino)."""
    return Classifier(RadDinoBackbone(pretrained), len(CLASSES))


def check_img_size(img_size):
    if img_size % PATCH:
        print(f"[warn] img_size {img_size} is not a multiple of the patch size {PATCH}; "
              f"the last {img_size % PATCH} pixel rows/columns are dropped by the patch embedding")


def _layer_id(name):
    import re
    m = re.search(r"\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)), "block"
    if "embeddings" in name:
        return None, "embed"
    return None, "top"  # final layernorm, head


def param_groups(model, args):
    """layer-wise lr decay + no weight decay on 1-D parameters / tokens."""
    named = list(model.named_parameters())
    n_layers = max(i for i, _ in (_layer_id(n) for n, _ in named) if i is not None) + 1
    groups = {}
    for n, p in named:
        i, kind = _layer_id(n)
        depth = i + 1 if kind == "block" else (0 if kind == "embed" else n_layers + 1)
        scale = LLRD ** (n_layers + 1 - depth)
        no_wd = WD_SKIP_NORM and (p.ndim <= 1 or n.endswith(".bias") or "token" in n)
        key = (round(scale, 8), no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": args.lr * scale, "weight_decay": 0.0 if no_wd else args.weight_decay,
                           "lr_scale": scale}
        groups[key]["params"].append(p)
    return sorted(groups.values(), key=lambda g: -g["lr_scale"])
# ============================ END MODEL-SPECIFIC ============================ #


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description=f"{MODEL_NAME}: CXR 4-class classification ({MODEL_DESC})",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode / io
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--data_dir", default=os.path.normpath(os.path.join(SCRIPT_DIR, "..", "cxr-dataset")),
                   help="folder with train/val/test _data csv files and _images folders")
    p.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "Output", MODEL_NAME))
    p.add_argument("--run_name", default=None,
                   help="sub-folder of output_dir; train default: <timestamp>; test default: "
                        "<weight folder>/test_<test csv name>")
    p.add_argument("--weight", nargs="+", default=None,
                   help="test: .pth file(s) or run folder(s) (all best_weight.pth below; softmax averaged). "
                        "train: optional initial weight")
    p.add_argument("--test_csv", default=None,
                   help="csv whose new_filename column gives the test images and order (default: "
                        "<data_dir>/test_data_sample.csv). Images are read from <data_dir>/<prefix>_images, "
                        "prefix = csv name before '_data' (test_data_sample.csv -> test_images, "
                        "val_data.csv -> val_images). If the csv has one-hot labels, metrics are reported")
    p.add_argument("--output_csv", default="test_data_sample.csv", help="file name of the one-hot prediction csv")
    p.add_argument("--covid_thr", default="auto",
                   help="COVID-19 threshold: predict COVID-19 only if p(COVID-19) >= t, else argmax of the other "
                        "classes; written as <output_csv>_thr.csv next to the argmax output_csv. "
                        "auto: t chosen on the out-of-fold probabilities (cv_folds >= 2; in test mode from the "
                        "fold<k>/val_probs.csv of the given run folder); a number: fixed t; none: off")
    # cross-validation
    p.add_argument("--cv_folds", type=int, default=0,
                   help="K >= 2: K-fold stratified CV over train+val merged, all folds trained in turn; "
                        "0/1: original train/val split")
    p.add_argument("--cv_seed", type=int, default=0, help="random_state of the StratifiedKFold split")
    # input / augmentation (train only, except img_size / fit)
    p.add_argument("--img_size", type=int, default=DEFAULTS["img_size"])
    p.add_argument("--fit", choices=["stretch", "letterbox"], default="stretch",
                   help="stretch: resize both sides to img_size; letterbox: keep the aspect ratio, pad black")
    p.add_argument("--rotation", type=float, default=0.0, help="random rotation in +-deg (0 = off)")
    p.add_argument("--brightness", type=float, default=0.0, help="random brightness factor in [1-b, 1+b] (0 = off)")
    p.add_argument("--contrast", type=float, default=0.0, help="random contrast factor in [1-c, 1+c] (0 = off)")
    p.add_argument("--resize_scale", type=float, default=1.0,
                   help="random resize strength s (1.0 = off); crop: RandomResizedCrop area in [s, 1]; "
                        "pad: image shrunk by a linear factor in [s, 1], black border")
    p.add_argument("--resize_mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--hflip", "--hfilp", dest="hflip", type=float, default=0.0,
                   help="horizontal flip probability (0 = off)")
    # optimisation
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--loss", choices=["ce", "focal"], default="ce",
                   help=f"cross-entropy or focal loss (gamma {FOCAL_GAMMA}); label smoothing {LABEL_SMOOTHING}")
    p.add_argument("--class_weight", choices=["none", "inverse", "sqrt"], default="sqrt",
                   help="loss class weights: inverse class frequency or its square root")
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                   help="early stopping on val macro-F1 (0 = off: train all epochs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class InfoLog:
    """print + append to a .txt file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").close()

    def __call__(self, msg=""):
        print(msg)
        sys.stdout.flush()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Letterbox:
    """long side scaled to `size` (aspect ratio kept), centred on a black size x size canvas."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR, reducing_gap=None)
        canvas = Image.new(img.mode, (self.size, self.size), 0)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"Letterbox({self.size})"


def fit_op(img_size, fit):
    return Letterbox(img_size) if fit == "letterbox" else transforms.Resize((img_size, img_size))


def build_transforms(args, train):
    """eval: fit -> tensor -> normalise. train: colour jitter (before the geometric ops, so borders stay black)
    -> fit -> random resize (crop / pad) -> rotation -> hflip -> tensor -> normalise."""
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not train:
        return transforms.Compose([fit_op(args.img_size, args.fit)] + norm)
    bil = transforms.InterpolationMode.BILINEAR
    t = []
    if args.brightness > 0:
        t.append(transforms.ColorJitter(brightness=args.brightness))
    if args.contrast > 0:
        t.append(transforms.ColorJitter(contrast=args.contrast))
    t.append(fit_op(args.img_size, args.fit))
    if args.resize_scale < 1.0:
        if args.resize_mode == "pad":  # zoom out only: the whole image stays visible
            t.append(transforms.RandomAffine(degrees=0, scale=(args.resize_scale, 1.0), fill=0, interpolation=bil))
        else:  # crop part of the fitted image and resize back
            t.append(transforms.RandomResizedCrop(args.img_size, scale=(args.resize_scale, 1.0), ratio=(0.9, 1.1)))
    if args.rotation > 0:
        t.append(transforms.RandomRotation(args.rotation, interpolation=bil))
    if args.hflip > 0:
        t.append(transforms.RandomHorizontalFlip(args.hflip))
    return transforms.Compose(t + norm)


class CXRDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L").convert("RGB")  # grey copied to 3 channels
        if self.transform:
            img = self.transform(img)
        return img, (-1 if self.labels is None else int(self.labels[i]))


def read_split(data_dir, split):
    """-> dict(files, paths, labels, source) of train_data.csv / val_data.csv."""
    df = pd.read_csv(os.path.join(data_dir, f"{split}_data.csv"))
    files = df["new_filename"].tolist()
    return {"files": files, "paths": [os.path.join(data_dir, f"{split}_images", f) for f in files],
            "labels": df[CLASSES].values.argmax(1).tolist(), "source": [split] * len(files)}


def subset(split, idx):
    return {k: [v[i] for i in idx] for k, v in split.items()}


def merge(a, b):
    return {k: a[k] + b[k] for k in a}


def read_test(args):
    """-> (files, paths, labels or None) from --test_csv."""
    csv = args.test_csv or os.path.join(args.data_dir, "test_data_sample.csv")
    df = pd.read_csv(csv)
    files = df["new_filename"].tolist()
    prefix = os.path.basename(csv).split("_data")[0]
    img_dir = os.path.join(args.data_dir, f"{prefix}_images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(args.data_dir, "test_images")
    labels = None
    if all(c in df for c in CLASSES):
        y = df[CLASSES].values
        if not pd.isna(y).any() and (y.astype(float).sum(1) == 1).all():
            labels = y.argmax(1).tolist()
    return csv, img_dir, files, [os.path.join(img_dir, f) for f in files], labels


def make_loader(ds, args, shuffle):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=False)


# --------------------------------------------------------------------------- #
# Loss / optimiser
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)
        ce = F.nll_loss(logp, target, reduction="none")
        if self.ls > 0:
            ce = (1 - self.ls) * ce - self.ls * logp.mean(1)
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        loss = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def class_weights(labels, mode):
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_optimizer(model, args):
    params = param_groups(model, args)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=SGD_MOMENTUM, weight_decay=args.weight_decay,
                               nesterov=True)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, epochs, steps_per_epoch):
    total, warm = epochs * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min((step - warm) / max(1, total - warm), 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, scheduler=None):
    """-> (metrics {loss, acc, f1}, probs (N, K), true (N,))."""
    train = optimizer is not None
    model.train(train)
    total, n, probs, trues = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=USE_AMP and device.type == "cuda"):
            logits = model(x).float()
            loss = criterion(logits, y) if y.min() >= 0 else torch.zeros((), device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        probs.append(logits.detach().softmax(1).cpu())
        trues.append(y.cpu())
    prob, true = torch.cat(probs).numpy(), torch.cat(trues).numpy()
    m = {"loss": total / max(n, 1)}
    if true.min() >= 0:
        pred = prob.argmax(1)
        m["acc"] = float((pred == true).mean())
        m["f1"] = float(f1_score(true, pred, average="macro", labels=range(len(CLASSES)), zero_division=0))
    return m, prob, true


def plot_curves(hist, best_epoch, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, name in zip(axes, ["loss", "f1"], ["Loss", "Macro-F1"]):
        ax.plot(hist["epoch"], hist[f"train_{key}"], marker="o", ms=3, label="train")
        ax.plot(hist["epoch"], hist[f"val_{key}"], marker="o", ms=3, label="validation")
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch ({best_epoch})")
        ax.set_title(f"{name} -- {title}")
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(true, pred, path, title):
    cm = confusion_matrix(true, pred, labels=range(len(CLASSES)))
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def metrics_text(true, pred, name):
    """macro-F1 / accuracy line + classification report + confusion matrix as text."""
    f1 = f1_score(true, pred, average="macro", labels=range(len(CLASSES)), zero_division=0)
    acc = float((np.asarray(true) == np.asarray(pred)).mean())
    rep = classification_report(true, pred, labels=range(len(CLASSES)), target_names=CLASSES, digits=4,
                                zero_division=0)
    cm = confusion_matrix(true, pred, labels=range(len(CLASSES)))
    return f1, acc, (f"{name}: macro-F1={f1:.4f} acc={acc:.4f} (n={len(true)})\n\n{rep}\n"
                     f"confusion matrix (rows=true, cols=pred, order={CLASSES}):\n{cm}")


# --------------------------------------------------------------------------- #
# COVID-19 threshold
# --------------------------------------------------------------------------- #
def macro_f1(true, pred):
    return float(f1_score(true, pred, average="macro", labels=range(len(CLASSES)), zero_division=0))


def thr_predict(P, t):
    """P: (N, K) probabilities. p(COVID-19) < t -> argmax over the other classes; t = 0 -> plain argmax."""
    pred = P.argmax(1)
    low = P[:, COVID] < t
    if low.any():
        Q = P[low].copy()
        Q[:, COVID] = -1.0
        pred[low] = Q.argmax(1)
    return pred


def select_thr(P, true):
    """best t on THR_GRID by macro-F1 on (P, true); ties -> the tied grid value closest to their median."""
    f = np.array([macro_f1(true, thr_predict(P, t)) for t in THR_GRID])
    best = THR_GRID[np.isclose(f, f.max())]
    return float(best[np.argmin(np.abs(best - np.median(best)))]), float(f.max())


def nested_thr(oof):
    """oof: DataFrame with fold, label and class columns. -> (nested predictions, {fold: t}, t_all, f1_all).
    nested: the threshold of each fold is chosen on the other folds (unbiased OOF estimate);
    t_all: chosen on all OOF images (used for the test submission; its OOF F1 is in-sample / optimistic)."""
    P, true, fold = oof[CLASSES].values, oof["label"].values.astype(int), oof["fold"].values
    pred, t_fold = np.empty(len(oof), dtype=int), {}
    for k in pd.unique(fold):
        inner = fold != k
        t_fold[int(k)], _ = select_thr(P[inner], true[inner])
        pred[~inner] = thr_predict(P[~inner], t_fold[int(k)])
    t_all, f_all = select_thr(P, true)
    return pred, t_fold, t_all, f_all


def load_oof(root):
    """OOF probabilities of a CV run folder from its fold<k>/val_probs.csv (None if there are none)."""
    import re
    parts = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = re.fullmatch(r"fold(\d+)", d)
        f = os.path.join(root, d, "val_probs.csv")
        if m and os.path.isfile(f):
            parts.append(pd.read_csv(f).assign(fold=int(m.group(1))))
    return pd.concat(parts, ignore_index=True) if len(parts) >= 2 else None


def fixed_thr(args):
    """numeric --covid_thr -> float, auto / none -> None."""
    return None if args.covid_thr in ("auto", "none") else float(args.covid_thr)


def thr_csv_name(args):
    stem, ext = os.path.splitext(args.output_csv)
    return f"{stem}_thr{ext or '.csv'}"


def header(log, args, title):
    log(f"===== {MODEL_NAME} {title} =====")
    log(f"time      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"model     : {MODEL_DESC}")
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"env       : torch {torch.__version__}, device {dev}")
    log(f"arguments : {json.dumps(vars(args), ensure_ascii=False)}")
    log(f"fixed     : {json.dumps(FIXED_INFO)}")
    log(f"            label_smoothing={LABEL_SMOOTHING}, focal_gamma={FOCAL_GAMMA}, "
        f"scheduler=warmup {WARMUP_EPOCHS} epoch + cosine (per step), amp={USE_AMP}")


def write_test_outputs(out_dir, args, files, prob, labels, log, t0, thr=None, thr_note=""):
    """one-hot csv (output_csv, argmax) + test_probs.csv (+ metrics if labels are known);
    thr: also <output_csv>_thr.csv with the COVID-19 threshold."""
    def write_csv(pred, name):
        df = pd.DataFrame(np.eye(len(CLASSES), dtype=int)[pred], columns=CLASSES)
        df.insert(0, "new_filename", files)
        df.to_csv(os.path.join(out_dir, name), index=False)

    def report(pred, tag, png):
        counts = np.bincount(pred, minlength=len(CLASSES))
        log(f"predicted : [{tag}] " + ", ".join(f"{c}={k} ({k / max(len(files), 1):.1%})"
                                                for c, k in zip(CLASSES, counts)))
        if labels is not None:
            f1, _, text = metrics_text(labels, pred, f"test [{tag}]")
            log("")
            log(text)
            log("")
            plot_confusion(labels, pred, os.path.join(out_dir, png), f"{MODEL_NAME} test {tag} (macro-F1={f1:.4f})")

    pred = prob.argmax(1)
    write_csv(pred, args.output_csv)
    dfp = pd.DataFrame(prob, columns=CLASSES)
    dfp.insert(0, "new_filename", files)
    dfp.to_csv(os.path.join(out_dir, "test_probs.csv"), index=False)
    log(f"images    : {len(files)}")
    report(pred, "argmax", "confusion_matrix_test.png")
    outputs = [args.output_csv]
    if thr is not None:
        pred_t = thr_predict(prob, thr)
        write_csv(pred_t, thr_csv_name(args))
        log(f"COVID thr : t={thr:.2f}{thr_note} -> {int((pred_t != pred).sum())} of {len(files)} predictions "
            f"changed")
        report(pred_t, f"COVID thr {thr:.2f}", "confusion_matrix_test_thr.png")
        outputs.append(thr_csv_name(args))
    log(f"outputs   : {', '.join(os.path.join(out_dir, o) for o in outputs)}, test_probs.csv")
    log(f"elapsed   : {time.time() - t0:.0f}s")


def save_checkpoint(model, args, epoch, val_f1, path):
    cfg = {"model": MODEL_NAME, "img_size": args.img_size, "fit": args.fit, "classes": CLASSES,
           "norm_mean": MEAN, "norm_std": STD, "args": vars(args)}
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_f1": val_f1}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("config", {}), ckpt
    return ckpt, {}, {}


def train_one(args, device, tr, va, test, out_dir, title):
    """train on tr, select by val macro-F1 on va, save outputs in out_dir. -> result dict."""
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    log = InfoLog(os.path.join(out_dir, "training_info.txt"))
    header(log, args, f"training ({title})")
    set_seed(args.seed)
    check_img_size(args.img_size)

    model = build_model(pretrained=True)
    if args.weight:
        sd, _, _ = load_checkpoint(args.weight[0])
        log(f"init      : {args.weight[0]} -> {model.load_state_dict(sd, strict=False)}")
    model.to(device)
    log(f"params    : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_loader = make_loader(CXRDataset(tr["paths"], tr["labels"], build_transforms(args, True)), args, True)
    val_loader = make_loader(CXRDataset(va["paths"], va["labels"], build_transforms(args, False)), args, False)
    log(f"data      : train={len(tr['files'])} {np.bincount(tr['labels'], minlength=len(CLASSES)).tolist()} | "
        f"val={len(va['files'])} {np.bincount(va['labels'], minlength=len(CLASSES)).tolist()} "
        f"(counts in order {CLASSES})")
    log(f"train tf  : {build_transforms(args, True)}".replace("\n", " "))

    w = class_weights(tr["labels"], args.class_weight)
    if w is not None:
        log(f"class w   : {dict(zip(CLASSES, w.numpy().round(3).tolist()))}")
        w = w.to(device)
    if args.loss == "focal":
        criterion = FocalLoss(w, FOCAL_GAMMA, LABEL_SMOOTHING).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()  # unweighted, comparable val loss
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    log(f"optimizer : {args.optimizer} lr={args.lr} wd={args.weight_decay}, {len(optimizer.param_groups)} param "
        f"group(s), {len(train_loader)} steps/epoch")
    log("")

    history, best = [], {"f1": -1.0, "loss": float("inf"), "epoch": 0}
    best_path = os.path.join(out_dir, "best_weight.pth")
    bad = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_m, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler, scheduler)
        va_m, _, _ = run_epoch(model, val_loader, device, eval_criterion)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in tr_m.items()}, **{f"val_{k}": v for k, v in va_m.items()}})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        improved = va_m["f1"] > best["f1"] or (va_m["f1"] == best["f1"] and va_m["loss"] < best["loss"])
        if improved:
            best, bad = {"f1": va_m["f1"], "loss": va_m["loss"], "epoch": epoch}, 0
            save_checkpoint(model, args, epoch, va_m["f1"], best_path)
        else:
            bad += 1
        log(f"[ep {epoch:3d}/{args.epochs}] train loss={tr_m['loss']:.4f} acc={tr_m['acc']:.4f} "
            f"f1={tr_m['f1']:.4f} | val loss={va_m['loss']:.4f} acc={va_m['acc']:.4f} f1={va_m['f1']:.4f} | "
            f"{time.time() - t0:.0f}s" + (" *best*" if improved else ""))
        if args.patience and bad >= args.patience:
            log(f"[early stop] no val macro-F1 improvement for {args.patience} epochs")
            break
    hist = pd.DataFrame(history)
    plot_curves(hist, best["epoch"], os.path.join(out_dir, "curves.png"), f"{MODEL_NAME} {title}")

    # best weight -> validation report + confusion matrix
    sd, _, _ = load_checkpoint(best_path)
    model.load_state_dict(sd)
    va_m, va_prob, va_true = run_epoch(model, val_loader, device, eval_criterion)
    va_pred = va_prob.argmax(1)
    plot_confusion(va_true, va_pred, os.path.join(out_dir, "confusion_matrix_best_val.png"),
                   f"{MODEL_NAME} {title}: best val (epoch {best['epoch']}, macro-F1={va_m['f1']:.4f})")
    val_df = pd.DataFrame(va_prob, columns=CLASSES)
    val_df.insert(0, "source", va["source"])
    val_df.insert(0, "label", va_true)
    val_df.insert(0, "new_filename", va["files"])
    val_df.to_csv(os.path.join(out_dir, "val_probs.csv"), index=False)
    _, _, text = metrics_text(va_true, va_pred, f"best val (epoch {best['epoch']})")
    log("")
    log(text)
    log("")
    log(f"best epoch: {best['epoch']}  val macro-F1={best['f1']:.4f}  val loss={best['loss']:.4f}")
    log(f"weight    : {best_path}")
    log(f"elapsed   : {time.time() - t_start:.0f}s")

    # test with the best weight
    t0 = time.time()
    tlog = InfoLog(os.path.join(out_dir, "testing_info.txt"))
    header(tlog, args, f"testing ({title})")
    csv, img_dir, files, paths, labels = test
    tlog(f"weight    : {best_path} (epoch {best['epoch']}, val macro-F1 {best['f1']:.4f})")
    tlog(f"test csv  : {csv} (images: {img_dir})")
    te_loader = make_loader(CXRDataset(paths, None, build_transforms(args, False)), args, False)
    _, te_prob, _ = run_epoch(model, te_loader, device, eval_criterion)
    # COVID threshold here only if fixed (--covid_thr <t>); auto is chosen on the OOF of the CV run
    write_test_outputs(out_dir, args, files, te_prob, labels, tlog, t0, thr=fixed_thr(args), thr_note=" (fixed)")

    del train_loader, val_loader, te_loader, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"title": title, "best_epoch": best["epoch"], "best_f1": best["f1"], "val_df": val_df,
            "test_prob": te_prob, "epochs_run": len(history)}


def train_mode(args, device):
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.output_dir, run_name)
    os.makedirs(root, exist_ok=True)
    test = read_test(args)
    tr, va = read_split(args.data_dir, "train"), read_split(args.data_dir, "val")
    if args.cv_folds < 2:
        train_one(args, device, tr, va, test, root, "train/val split")
        return

    t_start = time.time()
    data = merge(tr, va)
    skf = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.cv_seed)
    results = []
    for k, (ti, vi) in enumerate(skf.split(np.zeros(len(data["labels"])), data["labels"])):
        results.append(train_one(args, device, subset(data, ti), subset(data, vi), test,
                                 os.path.join(root, f"fold{k}"), f"fold {k}/{args.cv_folds}"))

    # out-of-fold summary (best-epoch weights of every fold)
    log = InfoLog(os.path.join(root, "training_info.txt"))
    header(log, args, f"{args.cv_folds}-fold CV summary")
    log(f"data      : train+val merged = {len(data['labels'])} images, StratifiedKFold(seed {args.cv_seed})")
    log("")
    for k, r in enumerate(results):
        log(f"fold {k}: best epoch {r['best_epoch']:3d}/{r['epochs_run']}  val macro-F1={r['best_f1']:.4f}")
    f1s = np.array([r["best_f1"] for r in results])
    log(f"per-fold macro-F1: mean={f1s.mean():.4f} std={f1s.std(ddof=1):.4f}")
    oof = pd.concat([r["val_df"].assign(fold=k) for k, r in enumerate(results)], ignore_index=True)
    oof.to_csv(os.path.join(root, "oof_probs.csv"), index=False)
    true, pred = oof["label"].values, oof[CLASSES].values.argmax(1)
    f1, _, text = metrics_text(true, pred, "OOF (each fold's best-epoch weight)")
    log("")
    log(text)
    plot_confusion(true, pred, os.path.join(root, "confusion_matrix_oof.png"),
                   f"{MODEL_NAME} {args.cv_folds}-fold OOF (macro-F1={f1:.4f})")
    thr, thr_note = oof_threshold(args, oof, log, root)
    log(f"elapsed   : {time.time() - t_start:.0f}s")

    # K-fold test ensemble (softmax average)
    t0 = time.time()
    tlog = InfoLog(os.path.join(root, "testing_info.txt"))
    header(tlog, args, f"testing ({args.cv_folds}-fold ensemble)")
    csv, img_dir, files, _, labels = test
    tlog(f"weights   : {[os.path.join(root, f'fold{k}', 'best_weight.pth') for k in range(args.cv_folds)]}")
    tlog(f"test csv  : {csv} (images: {img_dir}); softmax averaged over the folds")
    write_test_outputs(root, args, files, np.mean([r["test_prob"] for r in results], 0), labels, tlog, t0,
                       thr=thr, thr_note=thr_note)


def oof_threshold(args, oof, log, out_dir=None):
    """COVID-19 threshold from the OOF probabilities (--covid_thr auto) or fixed. Logs the OOF metrics with the
    threshold (auto: nested CV = unbiased) and, if out_dir, saves confusion_matrix_oof_thr.png.
    -> (t or None, note for the test log)."""
    if args.covid_thr == "none":
        return None, ""
    true = oof["label"].values.astype(int)
    log("")
    if args.covid_thr == "auto":
        pred, t_fold, t, f_all = nested_thr(oof)
        f1, _, text = metrics_text(true, pred, "OOF + COVID threshold (nested CV: each fold's t chosen on the "
                                               "other folds; unbiased)")
        log(f"COVID thr : per-fold t (chosen on the other folds) = {t_fold}")
        log(text)
        log("")
        log(f"COVID thr : t chosen on all OOF images = {t:.2f} (in-sample OOF macro-F1 {f_all:.4f}, optimistic) "
            f"-> used for the test submission")
        note, tag = " (chosen on the OOF)", "nested COVID thr"
    else:
        t = float(args.covid_thr)
        pred = thr_predict(oof[CLASSES].values, t)
        f1, _, text = metrics_text(true, pred, f"OOF + COVID threshold {t:.2f} (fixed)")
        log(text)
        note, tag = " (fixed)", f"COVID thr {t:.2f}"
    if out_dir:
        plot_confusion(true, pred, os.path.join(out_dir, "confusion_matrix_oof_thr.png"),
                       f"{MODEL_NAME} OOF, {tag} (macro-F1={f1:.4f})")
    return t, note


def expand_weights(items):
    out = []
    for w in items:
        if os.path.isdir(w):
            found = sorted(os.path.join(d, f) for d, _, fs in os.walk(w) for f in fs if f == "best_weight.pth")
            if not found:
                raise SystemExit(f"no best_weight.pth below {w}")
            out += found
        elif os.path.isfile(w):
            out.append(w)
        else:
            raise SystemExit(f"weight not found: {w}")
    return out


def test_mode(args, device):
    if not args.weight:
        raise SystemExit("--weight is required in test mode")
    t0 = time.time()
    weights = expand_weights(args.weight)
    csv, img_dir, files, paths, labels = read_test(args)
    if args.run_name:
        out_dir = os.path.join(args.output_dir, args.run_name)
    else:  # sub-folder next to the weight, so the train-mode test outputs are not overwritten
        base = (args.weight[0] if len(args.weight) == 1 and os.path.isdir(args.weight[0])
                else os.path.dirname(os.path.abspath(weights[0])))
        out_dir = os.path.join(base, "test_" + os.path.splitext(os.path.basename(csv))[0])
    os.makedirs(out_dir, exist_ok=True)
    log = InfoLog(os.path.join(out_dir, "testing_info.txt"))
    header(log, args, "testing")
    log(f"test csv  : {csv} (images: {img_dir})")
    probs = []
    for w in weights:
        sd, cfg, ck = load_checkpoint(w)
        classes = cfg.get("classes", CLASSES)
        if isinstance(classes, str):  # old main.py checkpoints: "normal,bacteria,..."
            classes = classes.split(",")
        if list(classes) != CLASSES:
            raise SystemExit(f"{w}: classes {classes} != {CLASSES}")
        # input size / fit come from the checkpoint
        args.img_size, args.fit = int(cfg.get("img_size", args.img_size)), cfg.get("fit", args.fit)
        model = build_model(pretrained=False)
        model.load_state_dict(sd)
        model.to(device)
        loader = make_loader(CXRDataset(paths, None, build_transforms(args, False)), args, False)
        _, p, _ = run_epoch(model, loader, device, nn.CrossEntropyLoss())
        probs.append(p)
        log(f"weight    : {w} (epoch {ck.get('epoch')}, val macro-F1 {ck.get('val_f1')}, "
            f"img_size {args.img_size}, fit {args.fit})")
        del loader, model
    if len(probs) > 1:
        log(f"ensemble  : softmax average of {len(probs)} weights")
    # COVID threshold: fixed, or (auto) chosen on the OOF of the given CV run folder
    thr, note = None, ""
    if args.covid_thr == "auto":
        oof = load_oof(args.weight[0]) if len(args.weight) == 1 else None
        if oof is None:
            log("COVID thr : auto -> off (needs one CV run folder with fold<k>/val_probs.csv as --weight)")
        else:
            log(f"COVID thr : OOF of {args.weight[0]} ({len(oof)} images, {oof['fold'].nunique()} folds)")
            thr, note = oof_threshold(args, oof, log)
    elif args.covid_thr != "none":
        thr, note = float(args.covid_thr), " (fixed)"
    write_test_outputs(out_dir, args, files, np.mean(probs, 0), labels, log, t0, thr=thr, thr_note=note)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "train":
        train_mode(args, device)
    else:
        test_mode(args, device)


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()
