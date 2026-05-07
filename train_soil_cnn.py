"""
train_soil_cnn.py  —  Krushi Mitra
====================================
Fine-tunes MobileNetV2 on the Phantom-fs 7-class soil dataset.

Dataset structure expected at DATASET_DIR:
    soil_dataset/
        Alluvial/   (or Alluvial Soil/)
        Black/
        Laterite/
        Red/
        Yellow/
        Arid/
        Mountain/

Usage:
    pip install torch torchvision Pillow
    python train_soil_cnn.py

Output:
    models/soil_cnn.pt   ← drop-in for soil_utils.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = "soil_dataset"        # root folder of the cloned dataset
MODEL_OUT     = "models/soil_cnn.pt"
NUM_CLASSES   = 7
EPOCHS        = 20
BATCH_SIZE    = 32
LR            = 1e-4
VAL_SPLIT     = 0.2
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def main():
    print(f"Using device: {DEVICE}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
    print(f"Classes found: {full_ds.classes}")
    print(f"Total images : {len(full_ds)}")

    n_val   = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # Apply val transform to val split
    val_ds.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Model: MobileNetV2 with frozen backbone ────────────────────────────────
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    # Unfreeze last 3 conv blocks for fine-tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    # ── Training setup ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / n_train * 100
        val_acc   = val_correct   / n_val   * 100

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"Train: {train_loss/n_train:.4f} ({train_acc:.1f}%)  "
              f"Val: {val_loss/n_val:.4f} ({val_acc:.1f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✓ Saved best model → {MODEL_OUT} (val_acc={val_acc:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_OUT}")
    print(f"\nClass order for CNN_CLASSES in soil_utils.py:")
    print(sorted(full_ds.classes))


if __name__ == "__main__":
    main()
