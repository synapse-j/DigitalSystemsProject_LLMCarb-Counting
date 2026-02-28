"""
pipeline/train_stage2.py
=========================
Trains a second ResNet-50 to estimate HOW MUCH food is in the image.
3-class output: small (0), medium (1), large (2).

Uses weighted loss to handle natural class imbalance
(most images look like medium portions).
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, STAGE2_MODEL_DIR, DEVICE,
    BATCH_SIZE, EPOCHS_S2, LR, NUM_WORKERS, NUM_PORTIONS,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# More aggressive augmentation than Stage 1 — simulates different
# camera distances and plate sizes to improve portion generalisation
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_data():
    base     = Path(DATASET_DIR) / "stage2"
    train_ds = datasets.ImageFolder(str(base / "train"), transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(str(base / "val"),   transform=VAL_TF)

    Path(STAGE2_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    idx_map = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(os.path.join(STAGE2_MODEL_DIR, "class_index.json"), "w") as f:
        json.dump(idx_map, f, indent=2)

    # Count per class for weighted loss
    class_counts = {}
    for _, label in train_ds.samples:
        cls = train_ds.classes[label]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    print(f"  Stage 2 — Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Portions: {class_counts}")
    return train_loader, val_loader, class_counts, train_ds.classes


def build_model() -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, NUM_PORTIONS),
    )
    return model


def make_criterion(class_counts, class_names):
    total   = sum(class_counts.values())
    weights = torch.tensor(
        [total / (len(class_counts) * class_counts.get(c, 1)) for c in class_names],
        dtype=torch.float32
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weights)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += images.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out  = model(images)
        loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += images.size(0)
    return loss_sum / total, correct / total


def train(epochs: int = EPOCHS_S2):
    train_loader, val_loader, class_counts, class_names = load_data()
    model     = build_model().to(DEVICE)
    criterion = make_criterion(class_counts, class_names)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    PHASE2    = max(1, epochs // 3)
    best_acc  = 0.0
    history   = []
    save_path = os.path.join(STAGE2_MODEL_DIR, "resnet50_stage2_best.pth")

    print(f"\n{'═'*60}")
    print(f"  Stage 2: Portion Size Estimator  ({epochs} epochs)")
    print(f"  Backbone unfreeze at epoch {PHASE2}")
    print(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        if epoch == PHASE2:
            print(f"\n  → Epoch {epoch}: Unfreezing layer4")
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
            optimizer = optim.Adam([
                {"params": model.fc.parameters(),     "lr": LR},
                {"params": model.layer4.parameters(), "lr": LR * 0.1},
            ])

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                         "train_acc": round(tr_acc, 4), "val_loss": round(vl_loss, 4),
                         "val_acc": round(vl_acc, 4)})

        flag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            flag = "  ← saved"

        print(f"  Ep {epoch:02d}/{epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.3f}  "
              f"({time.time()-t0:.1f}s){flag}")

    with open(os.path.join(STAGE2_MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✓ Best val accuracy: {best_acc:.3f}")
    print(f"  ✓ Saved to: {save_path}")
