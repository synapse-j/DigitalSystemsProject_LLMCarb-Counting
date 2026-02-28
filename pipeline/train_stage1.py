"""
pipeline/train_stage1.py
=========================
Trains ResNet-50 to identify WHAT food is in the image.
101-class classification using Food-101 categories.

Two-phase training:
  Phase 1 — only the new classification head is trained (backbone frozen)
  Phase 2 — top residual blocks unfrozen for fine-tuning
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
    DATASET_DIR, STAGE1_MODEL_DIR, DEVICE,
    BATCH_SIZE, EPOCHS_S1, LR, NUM_WORKERS,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
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
    base     = Path(DATASET_DIR) / "stage1"
    train_ds = datasets.ImageFolder(str(base / "train"), transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(str(base / "val"),   transform=VAL_TF)

    Path(STAGE1_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    idx_map = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(os.path.join(STAGE1_MODEL_DIR, "class_index.json"), "w") as f:
        json.dump(idx_map, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    print(f"  Stage 1 — Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"Classes: {len(train_ds.classes)}  Device: {DEVICE}")
    return train_loader, val_loader, len(train_ds.classes)


def build_model(n_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, n_classes),
    )
    return model


def unfreeze_top_layers(model):
    for name, param in model.named_parameters():
        if any(x in name for x in ("layer3", "layer4", "fc")):
            param.requires_grad = True


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


def train(epochs: int = EPOCHS_S1):
    train_loader, val_loader, n_classes = load_data()
    model     = build_model(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    PHASE2 = max(1, epochs // 4)
    best_acc  = 0.0
    history   = []
    save_path = os.path.join(STAGE1_MODEL_DIR, "resnet50_stage1_best.pth")

    print(f"\n{'═'*60}")
    print(f"  Stage 1: Food Category Classifier  ({epochs} epochs)")
    print(f"  Backbone unfreeze at epoch {PHASE2}")
    print(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        if epoch == PHASE2:
            print(f"\n  → Epoch {epoch}: Unfreezing layer3 + layer4")
            unfreeze_top_layers(model)
            optimizer = optim.Adam([
                {"params": model.fc.parameters(),     "lr": LR},
                {"params": model.layer3.parameters(), "lr": LR * 0.1},
                {"params": model.layer4.parameters(), "lr": LR * 0.1},
            ])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - PHASE2 + 1
            )

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

    with open(os.path.join(STAGE1_MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✓ Best val accuracy: {best_acc:.3f}")
    print(f"  ✓ Saved to: {save_path}")
