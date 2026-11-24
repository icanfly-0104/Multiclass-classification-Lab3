"""One-vs-Rest (OvR) CXR classification with four independent binary ResNet-18s.

The CSV is expected to contain one-hot labels for normal, bacteria, virus and
COVID-19. Each binary model learns one class vs. the other three. This file is
independent of main.py and does not modify it.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any


CLASSES = ("normal", "bacteria", "virus", "COVID-19")
CSV_COLUMNS = ("new_filename", *CLASSES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install with `pip install -r requirements.txt`."
        ) from exc
    return torch, (nn, DataLoader, Dataset), (Image, transforms), (models, plt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train four binary CXR classifiers with One-vs-Rest, or predict from a saved bundle."
    )
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--data-dir", type=Path, default=Path("nycu-aimi-2026-lab3-cxr"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_ovr_regularized"))
    parser.add_argument("--weights", type=Path, help="ovr_best_bundle.pth used by --mode predict")
    parser.add_argument("--test-dir", type=Path, help="Defaults to data-dir/test_images")
    parser.add_argument("--sample-csv", type=Path, help="Optional test CSV to preserve filename order")
    parser.add_argument("--prediction-csv", type=Path, help="Default: output-dir/test_data_sample.csv")
    parser.add_argument("--score-csv", type=Path, help="Default: output-dir/test_scores.csv")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Binary target smoothing in [0, 0.5)")
    parser.add_argument("--patience", type=int, default=7, help="Stop after this many epochs without global macro-F1 improvement; 0 disables")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--positive-weight-power", type=float, default=0.5,
                        help="pos_weight=(negative/positive)**power; 0 disables class weighting")
    parser.add_argument("--hard-class-dropout", type=float, default=0.3,
                        help="Dropout before the Bacteria and Virus output layer")
    parser.add_argument("--hard-class-weight-decay", type=float, default=1e-3,
                        help="AdamW weight decay for Bacteria and Virus models")
    parser.add_argument("--decision-rule", choices=("argmax", "threshold"), default="argmax",
                        help="Argmax compares raw OvR scores; threshold gates classes before choosing the highest score")
    parser.add_argument("--thresholds", type=float, nargs=4, metavar=("NORMAL", "BACTERIA", "VIRUS", "COVID"),
                        default=(0.5, 0.5, 0.5, 0.5), help="Per-class sigmoid thresholds for threshold decision rule")

    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--scale-range", type=float, default=0.1,
                        help="0.1 samples scaling from 0.9 to 1.1")
    parser.add_argument("--contrast-range", type=float, default=0.2)
    parser.add_argument("--brightness-range", type=float, default=0.0)
    return parser.parse_args()


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def read_labeled_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CSV_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(CSV_COLUMNS)}")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):
        try:
            bits = [int(row[name]) for name in CLASSES]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid labels in {path}:{line}; use one-hot 0/1 values") from exc
        if any(bit not in (0, 1) for bit in bits) or sum(bits) != 1:
            raise ValueError(f"Labels in {path}:{line} must have exactly one positive class")
        if not row.get("new_filename"):
            raise ValueError(f"Missing new_filename in {path}:{line}")
    return rows


def make_transforms(transforms: Any, image_size: int, training: bool, augmentation: bool,
                    rotation: float, scale: float, contrast: float, brightness: float) -> Any:
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if training and augmentation:
        if rotation:
            ops.append(transforms.RandomRotation(rotation))
        if scale:
            ops.append(transforms.RandomAffine(degrees=0, scale=(1.0 - scale, 1.0 + scale)))
        if contrast or brightness:
            ops.append(transforms.ColorJitter(contrast=contrast, brightness=brightness))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)


def make_dataset_class(torch: Any, Dataset: Any, Image: Any) -> Any:
    class CXRDataset(Dataset):
        def __init__(self, image_dir: Path, rows: list[dict[str, str]], transform: Any):
            self.rows = rows
            self.targets = [next(i for i, name in enumerate(CLASSES) if int(row[name]) == 1) for row in rows]
            available = {p.name.casefold(): p for p in image_dir.iterdir() if p.is_file()}
            missing = [row["new_filename"] for row in rows if row["new_filename"].casefold() not in available]
            if missing:
                raise FileNotFoundError(f"{len(missing)} listed images missing from {image_dir}; first: {missing[0]}")
            self.paths = [available[row["new_filename"].casefold()] for row in rows]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB")), torch.tensor(self.targets[index], dtype=torch.long)

    return CXRDataset


def binary_stats(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    pred = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, pred))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, pred))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, pred))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "positive_f1": positive_f1,
        "binary_macro_f1": (positive_f1 + negative_f1) / 2,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_pos = sum(label for _, label in ordered[index:end])
        group_neg = end - index - group_pos
        tp += group_pos
        fp += group_neg
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return ap


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    values = []
    for class_id in range(len(CLASSES)):
        tp = sum(y == class_id and p == class_id for y, p in zip(labels, predictions))
        fp = sum(y != class_id and p == class_id for y, p in zip(labels, predictions))
        fn = sum(y == class_id and p != class_id for y, p in zip(labels, predictions))
        denom = 2 * tp + fp + fn
        values.append(2 * tp / denom if denom else 0.0)
    return sum(values) / len(values)


def choose_classes(score_rows: list[list[float]], rule: str, thresholds: tuple[float, ...]) -> list[int]:
    predictions = []
    for scores in score_rows:
        if rule == "threshold":
            eligible = [i for i, score in enumerate(scores) if score >= thresholds[i]]
            if eligible:
                predictions.append(max(eligible, key=lambda i: scores[i]))
                continue
        predictions.append(max(range(len(CLASSES)), key=lambda i: scores[i]))
    return predictions


def run_binary_epoch(torch: Any, model: Any, loader: Any, criterion: Any, device: Any,
                     class_id: int, optimizer: Any | None = None, smoothing: float = 0.0
                     ) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n = 0
    labels: list[int] = []
    scores: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, multiclass_targets in loader:
            images = images.to(device)
            binary_targets = (multiclass_targets == class_id).to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            smoothed = binary_targets * (1.0 - smoothing) + (1.0 - binary_targets) * smoothing
            loss = criterion(logits, smoothed)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = binary_targets.size(0)
            total_loss += float(loss.item()) * batch_size
            n += batch_size
            labels.extend(binary_targets.detach().cpu().to(torch.int64).tolist())
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return total_loss / max(n, 1), labels, scores


def cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_resnet_model(models: Any, nn: Any, weights: Any, dropout: float) -> Any:
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    else:
        model.fc = nn.Linear(in_features, 1)
    return model


def plot_training(history: list[dict[str, float]], epoch_history: list[dict[str, float]],
                 output_dir: Path, plt: Any) -> None:
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(12, 4 * len(CLASSES)))
    for class_id, name in enumerate(CLASSES):
        rows = [row for row in history if row["class"] == name]
        epochs = [row["epoch"] for row in rows]
        axes[class_id, 0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
        axes[class_id, 0].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
        axes[class_id, 0].set(title=f"{name}: BCE loss", xlabel="Epoch", ylabel="Loss")
        axes[class_id, 0].legend()
        axes[class_id, 1].plot(epochs, [row["train_accuracy"] for row in rows], label="Train")
        axes[class_id, 1].plot(epochs, [row["val_accuracy"] for row in rows], label="Validation")
        axes[class_id, 1].set(title=f"{name}: binary accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[class_id, 1].set_ylim(0, 1)
        axes[class_id, 1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ovr_training_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for class_id, name in enumerate(CLASSES):
        rows = [row for row in history if row["class"] == name]
        axes.flat[class_id].plot([row["epoch"] for row in rows], [row["val_positive_f1"] for row in rows])
        axes.flat[class_id].set(title=f"{name}: validation positive F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "ovr_f1_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in epoch_history],
              [row["validation_macro_f1"] for row in epoch_history], marker="o")
    axis.set(title="Combined validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "ovr_macro_f1_curve.png", dpi=160)
    plt.close(fig)


def plot_roc_pr(labels_by_class: list[list[int]], scores_by_class: list[list[float]], output_dir: Path,
                plt: Any) -> None:
    roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
    pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
    for name, labels, scores in zip(CLASSES, labels_by_class, scores_by_class):
        positives = sum(labels)
        negatives = len(labels) - positives
        if not positives or not negatives:
            continue
        ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
        tpr = [0.0]
        fpr = [0.0]
        recall = [0.0]
        precision = [1.0]
        tp = fp = 0
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            group_positive = sum(label for _, label in ordered[i:j])
            group_negative = j - i - group_positive
            tp += group_positive
            fp += group_negative
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
            recall.append(tp / positives)
            precision.append(tp / (tp + fp))
            i = j
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc(labels, scores):.3f})")
        pr_ax.plot(recall, precision, label=f"{name} (AP={average_precision(labels, scores):.3f})")
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    roc_ax.set(title="Validation ROC: one-vs-rest", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    roc_ax.legend()
    roc_fig.tight_layout()
    roc_fig.savefig(output_dir / "validation_roc_curves.png", dpi=160)
    plt.close(roc_fig)
    pr_ax.set(title="Validation precision-recall: one-vs-rest", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    pr_ax.legend()
    pr_fig.tight_layout()
    pr_fig.savefig(output_dir / "validation_pr_curves.png", dpi=160)
    plt.close(pr_fig)


def plot_confusion_matrix(confusion: list[list[int]], output_path: Path, plt: Any) -> None:
    row_totals = [sum(row) for row in confusion]
    normalized = [
        [value / total if total else 0.0 for value in row]
        for row, total in zip(confusion, row_totals)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    matrices = (confusion, normalized)
    titles = ("Validation confusion matrix (counts)", "Validation confusion matrix (row-normalized)")
    for axis, matrix, title, is_normalized in zip(axes, matrices, titles, (False, True)):
        image = axis.imshow(matrix, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
        axis.set_yticks(range(len(CLASSES)), CLASSES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(title)
        threshold = max(max(row) for row in matrix) * 0.55
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                text = f"{value:.1%}" if is_normalized else f"{value}\n({normalized[i][j]:.1%})"
                color = "white" if value > threshold else "black"
                axis.text(j, i, text, ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_validation_analysis(args: argparse.Namespace, rows: list[dict[str, str]], targets: list[int],
                              score_rows: list[list[float]], output_dir: Path, plt: Any) -> dict[str, Any]:
    predictions = choose_classes(score_rows, args.decision_rule, tuple(args.thresholds))
    overall_f1 = macro_f1(targets, predictions)
    labels_by_class = [[int(target == class_id) for target in targets] for class_id in range(len(CLASSES))]
    scores_by_class = [[score_row[class_id] for score_row in score_rows] for class_id in range(len(CLASSES))]

    summary_fields = ["class", "positive_count", "negative_count", "threshold", "accuracy", "precision",
                      "recall", "specificity", "positive_f1", "binary_macro_f1", "balanced_accuracy",
                      "roc_auc", "average_precision", "tp", "fp", "tn", "fn"]
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for class_id, name in enumerate(CLASSES):
            labels = labels_by_class[class_id]
            scores = scores_by_class[class_id]
            stats = binary_stats(labels, scores, args.thresholds[class_id])
            writer.writerow({"class": name, "positive_count": sum(labels), "negative_count": len(labels) - sum(labels),
                             "threshold": args.thresholds[class_id], **stats,
                             "roc_auc": roc_auc(labels, scores), "average_precision": average_precision(labels, scores)})

    score_fields = ["new_filename", "true_class", "predicted_class", "correct", *[f"{name}_score" for name in CLASSES]]
    with (output_dir / "validation_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        for row, target, pred, scores in zip(rows, targets, predictions, score_rows):
            writer.writerow({"new_filename": row["new_filename"], "true_class": CLASSES[target],
                             "predicted_class": CLASSES[pred], "correct": int(target == pred),
                             **{f"{name}_score": score for name, score in zip(CLASSES, scores)}})

    confusion = [[sum(y == i and p == j for y, p in zip(targets, predictions))
                  for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *CLASSES])
        for name, values in zip(CLASSES, confusion):
            writer.writerow([name, *values])
    plot_confusion_matrix(confusion, output_dir / "confusion_matrix.png", plt)

    plot_roc_pr(labels_by_class, scores_by_class, output_dir, plt)
    return {"macro_f1": overall_f1, "accuracy": sum(y == p for y, p in zip(targets, predictions)) / max(len(targets), 1),
            "predictions": predictions, "confusion": confusion, "labels_by_class": labels_by_class,
            "scores_by_class": scores_by_class}


def train(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, Dataset: Any,
          Image: Any, transforms: Any, models: Any, plt: Any) -> None:
    train_rows = read_labeled_rows(args.data_dir / "train_data.csv")
    val_rows = read_labeled_rows(args.data_dir / "val_data.csv")
    augmentation = not args.no_augmentation
    dataset_class = make_dataset_class(torch, Dataset, Image)
    train_data = dataset_class(args.data_dir / "train_images", train_rows,
        make_transforms(transforms, args.image_size, True, augmentation, args.rotation_degrees,
                        args.scale_range, args.contrast_range, args.brightness_range))
    val_data = dataset_class(args.data_dir / "val_images", val_rows,
        make_transforms(transforms, args.image_size, False, False, args.rotation_degrees,
                        args.scale_range, args.contrast_range, args.brightness_range))

    seed_everything(torch, args.seed)
    device = get_device(torch, args.device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    counts = Counter(train_data.targets)
    models_by_class: list[Any] = []
    optimizers: list[Any] = []
    schedulers: list[Any] = []
    criteria: list[Any] = []
    positive_weights: list[float] = []
    model_weight_decays: list[float] = []
    model_dropouts: list[float] = []
    for class_id in range(len(CLASSES)):
        if counts[class_id] == 0:
            raise ValueError(f"No training positives found for class {CLASSES[class_id]}")
        neg = len(train_data.targets) - counts[class_id]
        pos_weight = (neg / counts[class_id]) ** args.positive_weight_power
        positive_weights.append(pos_weight)
        model_dropout = args.hard_class_dropout if CLASSES[class_id] in ("bacteria", "virus") else 0.0
        if args.no_pretrained:
            model = make_resnet_model(models, nn, None, model_dropout)
        else:
            try:
                model = make_resnet_model(models, nn, models.ResNet18_Weights.DEFAULT, model_dropout)
            except Exception as exc:
                raise SystemExit("Could not load pretrained ResNet-18; check network access or use --no-pretrained. " + str(exc)) from exc
        model.to(device)
        models_by_class.append(model)
        model_weight_decay = (args.hard_class_weight_decay
                              if CLASSES[class_id] in ("bacteria", "virus") else args.weight_decay)
        model_weight_decays.append(model_weight_decay)
        model_dropouts.append(model_dropout)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                            weight_decay=model_weight_decay))
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode="max", factor=0.5, patience=2))
        criteria.append(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    epoch_history: list[dict[str, float]] = []
    best_global_f1 = -1.0
    best_epoch = 0
    best_state: list[dict[str, Any]] | None = None
    best_val_scores: list[list[float]] | None = None
    no_improve = 0

    print(f"Training four OvR ResNet-18 models on {device}; train={len(train_data)}, validation={len(val_data)}")
    for epoch in range(1, args.epochs + 1):
        val_score_rows_by_model: list[list[float]] = []
        for class_id, name in enumerate(CLASSES):
            train_loss, train_labels, train_scores = run_binary_epoch(
                torch, models_by_class[class_id], train_loader, criteria[class_id], device,
                class_id, optimizers[class_id], args.label_smoothing)
            val_loss, val_labels, val_scores = run_binary_epoch(
                torch, models_by_class[class_id], val_loader, criteria[class_id], device,
                class_id, smoothing=args.label_smoothing)
            val_score_rows_by_model.append(val_scores)
            train_stats = binary_stats(train_labels, train_scores, 0.5)
            val_stats = binary_stats(val_labels, val_scores, args.thresholds[class_id])
            schedulers[class_id].step(val_stats["positive_f1"])
            history.append({"class": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "train_accuracy": train_stats["accuracy"], "val_accuracy": val_stats["accuracy"],
                            "train_positive_f1": train_stats["positive_f1"],
                            "val_positive_f1": val_stats["positive_f1"],
                            "val_roc_auc": roc_auc(val_labels, val_scores),
                            "val_average_precision": average_precision(val_labels, val_scores),
                            "pos_weight": positive_weights[class_id],
                            "weight_decay": model_weight_decays[class_id],
                            "dropout": model_dropouts[class_id]})

        score_rows = [[val_score_rows_by_model[k][i] for k in range(len(CLASSES))]
                      for i in range(len(val_data))]
        predictions = choose_classes(score_rows, args.decision_rule, tuple(args.thresholds))
        global_f1 = macro_f1(val_data.targets, predictions)
        accuracy = sum(y == p for y, p in zip(val_data.targets, predictions)) / len(val_data.targets)
        epoch_history.append({"epoch": epoch, "validation_accuracy": accuracy,
                              "validation_macro_f1": global_f1})
        print(f"Epoch {epoch:03d}/{args.epochs} | validation accuracy={accuracy:.4f}, multiclass macro-F1={global_f1:.4f} | "
              + " | ".join(f"{name} F1={history[-len(CLASSES)+i]['val_positive_f1']:.3f}"
                            for i, name in enumerate(CLASSES)))
        if global_f1 > best_global_f1:
            best_global_f1 = global_f1
            best_epoch = epoch
            best_state = [cpu_state_dict(model) for model in models_by_class]
            best_val_scores = score_rows
            no_improve = 0
        else:
            no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping: global validation macro-F1 did not improve for {args.patience} epochs")
            break

    if best_state is None or best_val_scores is None:
        raise RuntimeError("Training did not produce a checkpoint")
    for model, state in zip(models_by_class, best_state):
        model.load_state_dict(state)
    bundle_path = args.output_dir / "ovr_best_bundle.pth"
    config = {"classes": list(CLASSES), "image_size": args.image_size, "architecture": "resnet18_ovr",
              "decision_rule": args.decision_rule, "thresholds": list(args.thresholds),
              "dropout_by_class": {name: (args.hard_class_dropout if name in ("bacteria", "virus") else 0.0)
                                   for name in CLASSES},
              "best_epoch": best_epoch, "best_validation_macro_f1": best_global_f1}
    torch.save({"config": config, "model_state_dicts": best_state}, bundle_path)

    plot_training(history, epoch_history, args.output_dir, plt)
    with (args.output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(history[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "epoch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "validation_accuracy", "validation_macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(epoch_history)

    analysis = write_validation_analysis(args, val_rows, val_data.targets, best_val_scores, args.output_dir, plt)
    with (args.output_dir / "parameter_information.txt").open("w", encoding="utf-8") as handle:
        handle.write("OvR CXR training parameters and results\n")
        handle.write(f"architecture: four independent ResNet-18 binary classifiers\nclasses: {', '.join(CLASSES)}\n")
        handle.write(f"train_samples: {len(train_data)}\nvalidation_samples: {len(val_data)}\n")
        handle.write(f"train_class_counts: {dict(Counter(CLASSES[t] for t in train_data.targets))}\n")
        handle.write(f"validation_class_counts: {dict(Counter(CLASSES[t] for t in val_data.targets))}\n")
        handle.write(f"epochs_requested: {args.epochs}\nepochs_completed: {len(epoch_history)}\nbest_epoch: {best_epoch}\n")
        handle.write(f"batch_size: {args.batch_size}\nimage_size: {args.image_size}\nlearning_rate: {args.learning_rate}\n")
        handle.write(f"weight_decay: {args.weight_decay}\npositive_weight_power: {args.positive_weight_power}\n")
        handle.write(f"hard_class_regularization: Bacteria/Virus dropout={args.hard_class_dropout}, weight_decay={args.hard_class_weight_decay}\n")
        handle.write(f"positive_weights: {dict(zip(CLASSES, positive_weights))}\nlabel_smoothing: {args.label_smoothing}\n")
        handle.write(f"decision_rule: {args.decision_rule}\nthresholds: {dict(zip(CLASSES, args.thresholds))}\n")
        handle.write(f"pretrained: {not args.no_pretrained}\naugmentation_enabled: {augmentation}\n")
        handle.write(f"rotation_degrees: {args.rotation_degrees}\nscale_range: {args.scale_range}\n")
        handle.write(f"contrast_range: {args.contrast_range}\nbrightness_range: {args.brightness_range}\n")
        handle.write(f"seed: {args.seed}\ndevice: {device}\nbest_validation_macro_f1: {best_global_f1:.6f}\n")
        handle.write(f"best_validation_accuracy: {analysis['accuracy']:.6f}\ncheckpoint: {bundle_path}\n")
        handle.write("score_note: OvR sigmoid scores are independent and not guaranteed to sum to 1 or be calibrated probabilities.\n")
    print(f"Best bundle: {bundle_path} | epoch={best_epoch} | validation macro-F1={best_global_f1:.4f}")
    print(f"Analysis files: {args.output_dir}")


def read_test_names(image_dir: Path, sample_csv: Path | None) -> list[dict[str, str]]:
    if sample_csv and sample_csv.is_file():
        with sample_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "new_filename" not in reader.fieldnames:
                raise ValueError(f"{sample_csv} must contain a new_filename column")
            rows = [{"new_filename": row["new_filename"]} for row in reader]
    else:
        paths = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: p.name.casefold())
        rows = [{"new_filename": p.name} for p in paths]
    if not rows:
        raise ValueError(f"No test images listed by {sample_csv} or found in {image_dir}")
    available = {p.name.casefold() for p in image_dir.iterdir() if p.is_file()}
    missing = [row["new_filename"] for row in rows if row["new_filename"].casefold() not in available]
    if missing:
        raise FileNotFoundError(f"Test images listed in CSV are missing; first: {missing[0]}")
    return rows


def predict(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, Dataset: Any,
            Image: Any, transforms: Any, models: Any) -> None:
    if not args.weights or not args.weights.is_file():
        raise SystemExit("Prediction mode requires an existing --weights PATH to ovr_best_bundle.pth")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if tuple(config["classes"]) != CLASSES:
        raise ValueError("Checkpoint classes do not match this script's class order")
    image_dir = args.test_dir or (args.data_dir / "test_images")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    sample_path = args.sample_csv or (args.data_dir / "test_data_sample.csv")
    rows = read_test_names(image_dir, sample_path)

    class InferenceDataset(Dataset):
        def __init__(self) -> None:
            available = {p.name.casefold(): p for p in image_dir.iterdir() if p.is_file()}
            self.paths = [available[row["new_filename"].casefold()] for row in rows]
            self.transform = make_transforms(transforms, int(config["image_size"]), False, False,
                args.rotation_degrees, args.scale_range, args.contrast_range, args.brightness_range)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> Any:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB"))

    device = get_device(torch, args.device)
    model_list = []
    for state in checkpoint["model_state_dicts"]:
        class_id = len(model_list)
        dropout = float(config.get("dropout_by_class", {}).get(CLASSES[class_id], 0.0))
        model = make_resnet_model(models, nn, None, dropout)
        model.load_state_dict(state)
        model.to(device).eval()
        model_list.append(model)
    loader = DataLoader(InferenceDataset(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    score_rows: list[list[float]] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            columns = [torch.sigmoid(model(images).squeeze(1)).cpu().tolist() for model in model_list]
            score_rows.extend([[columns[class_id][i] for class_id in range(len(CLASSES))]
                               for i in range(len(columns[0]))])
    thresholds = tuple(float(value) for value in config["thresholds"])
    predicted = choose_classes(score_rows, config["decision_rule"], thresholds)

    prediction_path = args.prediction_csv or (args.output_dir / "test_data_sample.csv")
    score_path = args.score_csv or (args.output_dir / "test_scores.csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row, pred in zip(rows, predicted):
            writer.writerow({"new_filename": row["new_filename"],
                             **{name: int(i == pred) for i, name in enumerate(CLASSES)}})
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["new_filename", "predicted_class", *[f"{name}_score" for name in CLASSES]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, pred, scores in zip(rows, predicted, score_rows):
            writer.writerow({"new_filename": row["new_filename"], "predicted_class": CLASSES[pred],
                             **{f"{name}_score": score for name, score in zip(CLASSES, scores)}})
    print(f"Predictions: {prediction_path}\nOvR scores: {score_path}")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise SystemExit("--epochs, --batch-size, and --image-size must be positive")
    if not 0 <= args.label_smoothing < 0.5:
        raise SystemExit("--label-smoothing must be in [0, 0.5)")
    if args.positive_weight_power < 0:
        raise SystemExit("--positive-weight-power must be non-negative")
    if not 0 <= args.hard_class_dropout < 1:
        raise SystemExit("--hard-class-dropout must be in [0, 1)")
    if args.hard_class_weight_decay < 0:
        raise SystemExit("--hard-class-weight-decay must be non-negative")
    if args.rotation_degrees < 0 or not 0 <= args.scale_range < 1:
        raise SystemExit("--rotation-degrees must be non-negative and --scale-range must be in [0, 1)")
    if args.contrast_range < 0 or args.brightness_range < 0:
        raise SystemExit("--contrast-range and --brightness-range must be non-negative")
    if any(not 0 <= threshold <= 1 for threshold in args.thresholds):
        raise SystemExit("Each --thresholds value must be in [0, 1]")
    torch, (nn, DataLoader, Dataset), (Image, transforms), (models, plt) = dependencies()
    if args.mode == "train":
        train(args, torch, nn, DataLoader, Dataset, Image, transforms, models, plt)
    else:
        predict(args, torch, nn, DataLoader, Dataset, Image, transforms, models)


if __name__ == "__main__":
    main()
