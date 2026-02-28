"""
pipeline/cascade.py
====================
Loads both trained ResNet-50 models and runs the full cascade:

  Image → Stage 1 → food category
        → Stage 2 → portion size
        → Lookup  → carbohydrate range

Produces confusion matrices and all ML evaluation metrics.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, STAGE1_MODEL_DIR, STAGE2_MODEL_DIR,
    RESULTS_DIR, DEVICE, BATCH_SIZE, NUM_WORKERS,
    CARB_RANGE_LABELS, NUM_PORTIONS, lookup_carb_range,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFER_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

RANGE_LABELS = list(CARB_RANGE_LABELS.values())


# ──────────────────────────────────────────────
# MODEL LOADERS
# ──────────────────────────────────────────────

def load_stage1(n_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes),
    )
    ckpt = os.path.join(STAGE1_MODEL_DIR, "resnet50_stage1_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.eval().to(DEVICE)


def load_stage2() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_PORTIONS),
    )
    ckpt = os.path.join(STAGE2_MODEL_DIR, "resnet50_stage2_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.eval().to(DEVICE)


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class EvalDataset(Dataset):
    """Loads test images with carb range ground truth and food class name."""
    def __init__(self, root: str, transform):
        self.samples   = []
        self.transform = transform
        for range_dir in sorted(Path(root).iterdir()):
            if not range_dir.is_dir():
                continue
            label = int(range_dir.name.split("_")[1])
            for img in range_dir.glob("*.jpg"):
                food_class = "_".join(img.stem.split("_")[:-1])
                self.samples.append((str(img), label, food_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, food_class = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, food_class, path


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────

@torch.no_grad()
def run_cascade(stage1, stage2, loader, class_index: dict):
    true_labels, pred_labels = [], []
    detail = []

    for images, labels, food_classes, paths in loader:
        images = images.to(DEVICE)
        s1_preds = stage1(images).argmax(dim=1).cpu().numpy()
        s2_preds = stage2(images).argmax(dim=1).cpu().numpy()

        for i in range(len(labels)):
            food_pred    = class_index.get(str(s1_preds[i]), "unknown")
            portion_pred = int(s2_preds[i])
            final_range  = lookup_carb_range(food_pred, portion_pred)
            true_range   = labels[i].item()

            true_labels.append(true_range)
            pred_labels.append(final_range)
            detail.append({
                "image":          paths[i],
                "true_food":      food_classes[i],
                "pred_food":      food_pred,
                "pred_portion":   ["small", "medium", "large"][portion_pred],
                "true_range":     true_range,
                "pred_range":     final_range,
                "food_correct":   food_pred == food_classes[i],
            })

    return np.array(true_labels), np.array(pred_labels), detail


# ──────────────────────────────────────────────
# METRICS AND PLOTS
# ──────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, name: str, save_dir: str):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_n],
        [f"{name} — Counts", f"{name} — Normalised"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Range"); ax.set_ylabel("True Range")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(RANGE_LABELS, rotation=30, ha="right")
        ax.set_yticklabels(RANGE_LABELS)
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{data[i,j]:{fmt}}", ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=8)
    plt.suptitle(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"cm_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix → {path}")


def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    idx_path = os.path.join(STAGE1_MODEL_DIR, "class_index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError("Run train_stage1 first — class_index.json missing.")
    with open(idx_path) as f:
        class_index = json.load(f)

    print(f"\nLoading models onto {DEVICE}...")
    stage1 = load_stage1(len(class_index))
    stage2 = load_stage2()

    eval_dir = str(Path(DATASET_DIR) / "final_eval")
    ds       = EvalDataset(eval_dir, INFER_TF)
    loader   = DataLoader(ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
    print(f"  Test set: {len(ds):,} images")

    print("\nRunning cascade inference...")
    labels, preds, detail = run_cascade(stage1, stage2, loader, class_index)

    acc      = (labels == preds).mean()
    clin_acc = (np.abs(labels - preds) <= 1).mean()
    report   = classification_report(labels, preds, target_names=RANGE_LABELS,
                                     output_dict=True, zero_division=0)

    print(f"\n  Exact accuracy:                  {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Clinically acceptable (±1 rng):  {clin_acc:.3f}  ({clin_acc*100:.1f}%)")

    plot_confusion_matrix(labels, preds, "Cascaded ResNet-50", RESULTS_DIR)

    results = {
        "model":                                   "Cascaded ResNet-50",
        "test_accuracy":                           round(float(acc), 4),
        "clinically_acceptable_accuracy_±1_range": round(float(clin_acc), 4),
        "classification_report":                   report,
        "confusion_matrix":                        confusion_matrix(labels, preds).tolist(),
    }
    out_path = os.path.join(RESULTS_DIR, "cascade_results.json")
    with open(out_path, "w") as f:
        json.dump({"cascade": results, "detail": detail}, f, indent=2)
    print(f"  ✓ Results → {out_path}")
    return results
