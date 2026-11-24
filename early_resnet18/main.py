"""Train and run a four-class chest X-ray classifier.

Expected layout (the dataset directory may be selected with --data-dir)::

    nycu-aimi-2026-lab3-cxr/
        train_images/  val_images/  test_images/
        train_data.csv  val_data.csv

Install dependencies with ``pip install -r requirements.txt``. The default
model is an ImageNet-pretrained ResNet-18 fine-tuned for the four labels.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any


CLASSES = ("normal", "bacteria", "virus", "COVID-19")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_COLUMNS = ("new_filename", *CLASSES)


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
            f"Missing dependency: {exc}. Install the packages with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torch, (nn, DataLoader, Dataset), (Image, transforms), (models, plt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CXR classifier or predict labels for test images."
    )
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--data-dir", type=Path, default=Path("nycu-aimi-2026-lab3-cxr"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split-rate", type=float, default=0.05,
                        help="Validation fraction after combining train_data.csv and val_data.csv (stratified by class)")
    parser.add_argument("--weights", type=Path, help="Checkpoint used by --mode predict")
    parser.add_argument("--test-dir", type=Path, help="Override data-dir/test_images")
    parser.add_argument("--sample-csv", type=Path, help="Optional test CSV; row order and filenames are retained")
    parser.add_argument("--prediction-csv", type=Path, help="Prediction output path (default: output-dir/test_data_sample.csv)")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience; 0 disables it")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="Inverse-frequency class weight power; 0 disables weighting, 0.5 uses sqrt inverse frequency")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=10.0,
                        help="Maximum absolute random rotation in degrees")
    parser.add_argument("--scale-range", type=float, default=0.1,
                        help="Random scaling deviation; 0.05 samples scales from 0.95 to 1.05")
    parser.add_argument("--contrast-range", type=float, default=0.3,
                        help="Random contrast jitter strength; 0 disables it")
    parser.add_argument("--brightness-range", type=float, default=0.1,
                        help="Random brightness jitter strength; 0 disables it")
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
        raise SystemExit("CUDA was requested but is not available in this PyTorch environment.")
    return device


def read_labeled_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV was not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CSV_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: {', '.join(CSV_COLUMNS)}")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):
        try:
            bits = [int(row[name]) for name in CLASSES]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class labels at {csv_path}:{line}; expected 0/1 values") from exc
        if any(bit not in (0, 1) for bit in bits) or sum(bits) != 1:
            raise ValueError(f"Labels at {csv_path}:{line} must be one-hot (exactly one class is 1)")
        if not row.get("new_filename"):
            raise ValueError(f"Missing new_filename at {csv_path}:{line}")
    return rows


def make_transforms(transforms: Any, image_size: int, training: bool, augmentation: bool,
                    rotation_degrees: float = 0.0, scale_range: float = 0.00,
                    contrast_range: float = 0.0, brightness_range: float = 0.0) -> Any:
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if training and augmentation:
        if rotation_degrees:
            ops.append(transforms.RandomRotation(degrees=rotation_degrees))
        if scale_range:
            ops.append(transforms.RandomAffine(degrees=0, scale=(1.0 - scale_range, 1.0 + scale_range)))
        if brightness_range or contrast_range:
            ops.append(transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range))
    ops.extend([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)

def stratified_split(rows: list[dict[str, str]], validation_rate: float,
                     seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split the combined rows by class while keeping at least one sample in each split."""
    if not 0.0 < validation_rate < 1.0:
        raise ValueError("--split-rate must be greater than 0 and less than 1")
    grouped: dict[int, list[dict[str, str]]] = {class_id: [] for class_id in range(len(CLASSES))}
    for row in rows:
        class_id = next(i for i, name in enumerate(CLASSES) if int(row[name]) == 1)
        grouped[class_id].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for class_id, class_rows in grouped.items():
        if len(class_rows) < 2:
            raise ValueError(f"Cannot stratify class {CLASSES[class_id]} with fewer than 2 samples")
        rng.shuffle(class_rows)
        val_count = round(len(class_rows) * validation_rate)
        val_count = min(max(val_count, 1), len(class_rows) - 1)
        val_rows.extend(class_rows[:val_count])
        train_rows.extend(class_rows[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def make_dataset_class(torch: Any, Dataset: Any, Image: Any) -> Any:
    class CXRDataset(Dataset):
        def __init__(self, image_dirs: Path | list[Path], rows: list[dict[str, str]], transform: Any):
            if isinstance(image_dirs, Path):
                image_dirs = [image_dirs]
            self.rows = rows
            self.transform = transform
            self.targets = [next(i for i, name in enumerate(CLASSES) if int(row[name]) == 1) for row in rows]
            needed = {row["new_filename"].casefold() for row in rows}
            available: dict[str, Path] = {}
            for image_dir in image_dirs:
                for path in image_dir.iterdir():
                    if not path.is_file():
                        continue
                    key = path.name.casefold()
                    if key not in needed:
                        continue
                    if key in available and available[key] != path:
                        raise ValueError(f"Duplicate image filename exists in multiple split folders: {path.name}")
                    available[key] = path
            missing = [row["new_filename"] for row in rows if row["new_filename"].casefold() not in available]
            if missing:
                raise FileNotFoundError(f"{len(missing)} listed images are missing from {image_dirs}; first: {missing[0]}")
            self.paths = [available[row["new_filename"].casefold()] for row in rows]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            with Image.open(self.paths[index]) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
            return tensor, torch.tensor(self.targets[index], dtype=torch.long)

    return CXRDataset


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for class_id in range(len(CLASSES)):
        tp = sum(y == class_id and p == class_id for y, p in zip(labels, predictions))
        fp = sum(y != class_id and p == class_id for y, p in zip(labels, predictions))
        fn = sum(y == class_id and p != class_id for y, p in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def class_weights(torch: Any, targets: list[int], power: float, device: Any) -> Any:
    counts = Counter(targets)
    absent = [CLASSES[i] for i in range(len(CLASSES)) if counts[i] == 0]
    if absent:
        raise ValueError(f"Training data has no samples for: {', '.join(absent)}")
    weights = [(len(targets) / counts[i]) ** power for i in range(len(CLASSES))]
    mean = sum(weights) / len(weights)
    return torch.tensor([weight / mean for weight in weights], dtype=torch.float32, device=device)


def run_epoch(torch: Any, model: Any, loader: Any, criterion: Any, device: Any,
              optimizer: Any | None = None, extra_criterion: Any | None = None
              ) -> tuple[float, float, list[int], list[int], float | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_loss_denominator = 0.0
    total_extra_loss = 0.0
    total_extra_denominator = 0.0
    total = 0
    labels: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_loss_denominator = (
                float(criterion.weight[targets].sum().item())
                if criterion.weight is not None else float(targets.size(0))
            )
            total_loss += float(loss.item()) * batch_loss_denominator
            total_loss_denominator += batch_loss_denominator
            if extra_criterion is not None:
                extra_loss = extra_criterion(logits, targets)
                extra_denominator = (
                    float(extra_criterion.weight[targets].sum().item())
                    if extra_criterion.weight is not None else float(targets.size(0))
                )
                total_extra_loss += float(extra_loss.item()) * extra_denominator
                total_extra_denominator += extra_denominator
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            total += batch_size
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    extra_loss_value = (total_extra_loss / max(total_extra_denominator, 1.0)
                        if extra_criterion is not None else None)
    return (total_loss / max(total_loss_denominator, 1.0),
            sum(y == p for y, p in zip(labels, predictions)) / max(total, 1),
            labels, predictions, extra_loss_value)


def save_curves(history: list[dict[str, float]], accuracy_path: Path,
                f1_path: Path, plt: Any) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_accuracy"] for row in history], label="Training")
    axes[0].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_loss"] for row in history], label="Training")
    axes[1].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(accuracy_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [row["train_macro_f1"] for row in history], label="Training")
    axis.plot(epochs, [row["val_macro_f1"] for row in history], label="Validation")
    axis.set(title="Macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(f1_path, dpi=160)
    plt.close(figure)


def save_validation_loss_comparison(history: list[dict[str, float]], path: Path, plt: Any) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [row["val_weighted_loss"] for row in history], label="Weighted validation loss")
    axis.plot(epochs, [row["val_unweighted_loss"] for row in history], label="Unweighted validation loss")
    axis.set(title="Validation loss calculation comparison", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_confusion_matrix(labels: list[int], predictions: list[int], path: Path, plt: Any) -> None:
    """Save the raw confusion matrix for a validation prediction snapshot."""
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for target, prediction in zip(labels, predictions):
        matrix[target][prediction] += 1

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count")
    axis.set(
        title="Validation Confusion Matrix (Best Macro-F1)",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    axis.tick_params(axis="x", labelrotation=25)
    threshold = max((max(row) for row in matrix), default=0) / 2
    for true_class, row in enumerate(matrix):
        for predicted_class, count in enumerate(row):
            axis.text(predicted_class, true_class, str(count), ha="center", va="center",
                      color="white" if count > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, Dataset: Any,
          Image: Any, transforms: Any, models: Any, plt: Any) -> None:
    print("Get data")
    combined_rows: list[dict[str, str]] = []
    for source_split in ("train", "val"):
        source_rows = read_labeled_rows(args.data_dir / f"{source_split}_data.csv")
        combined_rows.extend({**row, "source_split": source_split} for row in source_rows)
    train_rows, val_rows = stratified_split(combined_rows, args.split_rate, args.seed)
    print(f"Merged {len(combined_rows)} labeled images; split-rate={args.split_rate:.3f}, "
          f"train={len(train_rows)}, validation={len(val_rows)}")
    
    augmentation = not args.no_augmentation
    dataset_class = make_dataset_class(torch, Dataset, Image)
    image_dirs = [args.data_dir / "train_images", args.data_dir / "val_images"]
    train_data = dataset_class(image_dirs, train_rows,
                               make_transforms(transforms, args.image_size, True, augmentation,
                                               args.contrast_range, args.brightness_range))
    val_data = dataset_class(image_dirs, val_rows,
                             make_transforms(transforms, args.image_size, False, False))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.device != "cpu")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.device != "cpu")


    device = get_device(torch, args.device)
    print("Set " + str(device) + " as device")

    print("Set " + str(args.seed) + " as random seed")
    seed_everything(torch, args.seed)
    if args.no_pretrained:
        print("Training from scratch")
        model = models.resnet18(weights=None)
    else:
        try:
            print("Training with pretrain weight")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as exc:
            raise SystemExit(
                "Could not load ImageNet pretrained ResNet-18 weights. Check network access, "
                "or rerun with --no-pretrained. Original error: " + str(exc)
            ) from exc
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model.to(device)
    weights = class_weights(torch, train_data.targets, args.class_weight_power, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    unweighted_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pth"
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("new_filename", "class", "original_split", "assigned_split"))
        writer.writeheader()
        for assigned_split, split_rows in (("train", train_rows), ("validation", val_rows)):
            for row in split_rows:
                class_name = next(name for name in CLASSES if int(row[name]) == 1)
                writer.writerow({"new_filename": row["new_filename"], "class": class_name,
                                 "original_split": row["source_split"], "assigned_split": assigned_split})
    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_val_labels: list[int] = []
    best_val_predictions: list[int] = []
    epochs_without_improvement = 0

    print("Training started")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_labels, train_predictions, _ = run_epoch(
            torch, model, train_loader, criterion, device, optimizer)
        
        val_loss, val_accuracy, val_labels, val_predictions, val_unweighted_loss = run_epoch(
            torch, model, val_loader, criterion, device, extra_criterion=unweighted_criterion)
        
        train_f1 = macro_f1(train_labels, train_predictions)
        val_f1 = macro_f1(val_labels, val_predictions)

        scheduler.step(val_f1)
        row = {"epoch": float(epoch), "train_loss": train_loss, "train_accuracy": train_accuracy,
               "train_macro_f1": train_f1, "val_loss": val_loss,
               "val_weighted_loss": val_loss, "val_unweighted_loss": float(val_unweighted_loss),
               "val_accuracy": val_accuracy, "val_macro_f1": val_f1}
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_accuracy:.4f}, macro-F1 {train_f1:.4f} | "
              f"val loss weighted {val_loss:.4f}, unweighted {val_unweighted_loss:.4f}, "
              f"acc {val_accuracy:.4f}, macro-F1 {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_val_labels = val_labels.copy()
            best_val_predictions = val_predictions.copy()
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "classes": list(CLASSES),
                        "image_size": args.image_size, "best_val_macro_f1": best_f1,
                        "epoch": epoch, "architecture": "resnet18", "split_rate": args.split_rate,
                        "split_seed": args.seed}, best_path)
        else:
            epochs_without_improvement += 1

        save_curves(history, args.output_dir / "training_curves.png",
                    args.output_dir / "macro_f1_curve.png", plt)
        
        save_validation_loss_comparison(history, args.output_dir / "validation_loss_comparison.png", plt)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping: validation macro-F1 did not improve for {args.patience} epochs.")
            break

    confusion_matrix_path = args.output_dir / "best_validation_confusion_matrix.png"
    save_confusion_matrix(best_val_labels, best_val_predictions, confusion_matrix_path, plt)

    parameter_lines = [
        "CXR classifier training parameters",
        f"architecture: resnet18",
        f"pretrained: {not args.no_pretrained}",
        f"classes: {', '.join(CLASSES)}",
        f"source_csvs_merged: {args.data_dir / 'train_data.csv'}, {args.data_dir / 'val_data.csv'}",
        f"stratified_split_rate: {args.split_rate}",
        f"split_seed: {args.seed}",
        f"split_manifest: {args.output_dir / 'split_manifest.csv'}",
        f"train_samples: {len(train_rows)}",
        f"validation_samples: {len(val_rows)}",
        f"train_class_counts: {dict(Counter(CLASSES[target] for target in train_data.targets))}",
        f"validation_class_counts: {dict(Counter(CLASSES[target] for target in val_data.targets))}",
        f"epochs_requested: {args.epochs}",
        f"epochs_completed: {len(history)}",
        f"best_epoch: {best_epoch}",
        f"batch_size: {args.batch_size}",
        f"num_workers: {args.num_workers}",
        f"image_size: {args.image_size}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"optimizer: AdamW",
        f"scheduler: ReduceLROnPlateau(mode=max, factor=0.5, patience=2)",
        f"label_smoothing: {args.label_smoothing}",
        f"class_weight_power: {args.class_weight_power}",
        "validation_loss_comparison: weighted vs unweighted cross-entropy on identical predictions; label smoothing is matched",
        f"early_stopping_patience: {args.patience}",
        f"augmentation_enabled: {augmentation}",
        f"random_rotation_degrees: {args.rotation_degrees if augmentation else 0}",
        f"random_scale_range: {(1.0 - args.scale_range, 1.0 + args.scale_range) if augmentation else 'disabled'}",
        f"random_contrast_range: {args.contrast_range if augmentation else 0}",
        f"random_brightness_range: {args.brightness_range if augmentation else 0}",
        f"seed: {args.seed}",
        f"device: {device}",
        f"best_validation_macro_f1: {best_f1:.6f}",
        f"best_validation_confusion_matrix: {confusion_matrix_path}",
        f"checkpoint: {best_path}",
    ]
    parameter_path = args.output_dir / "parameter_information.txt"
    parameter_path.write_text("\n".join(parameter_lines) + "\n", encoding="utf-8")
    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}; checkpoint: {best_path}")
    print(f"Curves: {args.output_dir / 'training_curves.png'}, {args.output_dir / 'macro_f1_curve.png'}, "
          f"{args.output_dir / 'validation_loss_comparison.png'}")
    print(f"Best-validation-F1 confusion matrix: {confusion_matrix_path}")
    print(f"Parameters: {parameter_path}")


def predict(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, Dataset: Any,
            Image: Any, transforms: Any, models: Any) -> None:
    if not args.weights:
        raise SystemExit("Prediction mode requires --weights PATH (for example outputs/best_model.pth).")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {args.weights}")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    classes = tuple(checkpoint.get("classes", CLASSES))
    if classes != CLASSES:
        raise ValueError(f"Checkpoint class order {classes} does not match expected class order {CLASSES}")
    image_size = int(checkpoint.get("image_size", args.image_size))
    image_dir = args.test_dir or (args.data_dir / "test_images")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory was not found: {image_dir}")

    sample_path = args.sample_csv or (args.data_dir / "test_data_sample.csv")
    if sample_path.is_file():
        with sample_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "new_filename" not in reader.fieldnames:
                raise ValueError(f"{sample_path} must contain a new_filename column")
            rows = [{"new_filename": row["new_filename"]} for row in reader]
    else:
        paths = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: p.name.casefold())
        rows = [{"new_filename": path.name} for path in paths]
    if not rows:
        raise ValueError(f"No test image names found in {sample_path} or {image_dir}")

    class InferenceDataset(Dataset):
        def __init__(self) -> None:
            available = {p.name.casefold(): p for p in image_dir.iterdir() if p.is_file()}
            missing = [row["new_filename"] for row in rows if row["new_filename"].casefold() not in available]
            if missing:
                raise FileNotFoundError(f"Test images listed in CSV are missing; first: {missing[0]}")
            self.paths = [available[row["new_filename"].casefold()] for row in rows]
            self.transform = make_transforms(transforms, image_size, False, False)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> Any:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB"))

    device = get_device(torch, args.device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(InferenceDataset(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    predicted: list[int] = []
    with torch.no_grad():
        for images in loader:
            predicted.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    output_path = args.prediction_csv or (args.output_dir / "test_data_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row, class_id in zip(rows, predicted):
            writer.writerow({"new_filename": row["new_filename"],
                             **{name: int(i == class_id) for i, name in enumerate(CLASSES)}})
    print(f"Predictions for {len(predicted)} images saved to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise SystemExit("--epochs, --batch-size, and --image-size must be positive")
    if not 0.0 < args.split_rate < 1.0:
        raise SystemExit("--split-rate must be greater than 0 and less than 1")
    if args.class_weight_power < 0:
        raise SystemExit("--class-weight-power must be non-negative")
    if args.rotation_degrees < 0 or not 0 <= args.scale_range < 1:
        raise SystemExit("--rotation-degrees must be non-negative and --scale-range must be in [0, 1)")
    if args.contrast_range < 0 or args.brightness_range < 0:
        raise SystemExit("--contrast-range and --brightness-range must be non-negative")
    torch, (nn, DataLoader, Dataset), (Image, transforms), (models, plt) = dependencies()
    if args.mode == "train":
        train(args, torch, nn, DataLoader, Dataset, Image, transforms, models, plt)
    else:
        predict(args, torch, nn, DataLoader, Dataset, Image, transforms, models)


if __name__ == "__main__":
    main()
