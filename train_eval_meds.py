import argparse
import os
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import timm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# -----------------------------
# Data loaders
# -----------------------------

def get_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict[int, str]]:
    """
    Returns train/val/test dataloaders, number of classes, and idx->class mapping.
    Expects:
      data_root/train
      data_root/val
      data_root/test
    """

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Common mean/std for ImageNet pretrained networks
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_classes = len(train_dataset.classes)
    idx_to_class = {i: c for i, c in enumerate(train_dataset.classes)}

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes, idx_to_class


# -----------------------------
# Model builders
# -----------------------------

def build_resnet(num_classes: int, variant: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet model (18 or 50) and adjust final layer.
    """
    if variant == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
    elif variant == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit(num_classes: int, variant: str = "vit_b16", pretrained: bool = True) -> nn.Module:
    """
    Build a ViT model using timm.
    Common variants:
      vit_b16 -> 'vit_base_patch16_224'
      vit_s16 -> 'vit_small_patch16_224'
    """
    if variant == "vit_b16":
        name = "vit_base_patch16_224"
    elif variant == "vit_s16":
        name = "vit_small_patch16_224"
    else:
        raise ValueError(f"Unsupported ViT variant: {variant}")

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model


# -----------------------------
# Training & evaluation loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns accuracy, y_true, y_pred.
    """
    model.eval()
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    acc = correct / total
    return acc, np.array(all_true), np.array(all_pred)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="medicine_boxes_split",
                        help="Root folder with train/val/test subdirectories")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_b16", "vit_s16"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="If set, do not use ImageNet-pretrained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes, idx_to_class = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Found {num_classes} classes:")
    for idx, name in idx_to_class.items():
        print(f"  {idx}: {name}")

    pretrained = not args.no_pretrained

    # Build model
    if args.model.startswith("resnet"):
        model = build_resnet(num_classes, variant=args.model, pretrained=pretrained)
    else:  # vit_b16 or vit_s16
        model = build_vit(num_classes, variant=args.model, pretrained=pretrained)

    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0
    best_state = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest val acc: {best_val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_acc, y_true, y_pred = test_model(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Confusion matrix & classification report
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()
