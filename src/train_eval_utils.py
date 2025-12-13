"""
学習・評価用ユーティリティ関数
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ==============================================================================
# 定数
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["day", "night", "sunrise_sunset"]


# ==============================================================================
# Transforms
# ==============================================================================
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    訓練用のTransformsを取得
    
    Args:
        image_size: 画像サイズ
    
    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    検証・テスト用のTransformsを取得
    
    Args:
        image_size: 画像サイズ
    
    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ==============================================================================
# データローダー
# ==============================================================================
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    DataLoaderを作成する
    
    Args:
        data_dir: データディレクトリ（train/val/testサブディレクトリを含む）
        batch_size: バッチサイズ
        image_size: 画像サイズ
        num_workers: データロードのワーカー数
    
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_path = Path(data_dir)
    
    # Transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Datasets
    train_dataset = datasets.ImageFolder(
        data_path / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        data_path / "val", transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        data_path / "test", transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names


# ==============================================================================
# モデル
# ==============================================================================
def create_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    ResNet18ベースのモデルを作成する
    
    Args:
        num_classes: クラス数
        pretrained: ImageNet事前学習済み重みを使用するか
    
    Returns:
        nn.Module
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    model = models.resnet18(weights=weights)
    
    # 最終全結合層を置換
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


# ==============================================================================
# 学習
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    1エポック分の学習を行う
    
    Args:
        model: モデル
        dataloader: DataLoader
        criterion: 損失関数
        optimizer: オプティマイザ
        device: デバイス
    
    Returns:
        (平均loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    検証を行う
    
    Args:
        model: モデル
        dataloader: DataLoader
        criterion: 損失関数
        device: デバイス
    
    Returns:
        (平均loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    モデルを学習する
    
    Args:
        model: モデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        criterion: 損失関数
        optimizer: オプティマイザ
        device: デバイス
        num_epochs: エポック数
        save_path: 最良モデルの保存パス
    
    Returns:
        学習履歴（train_loss, train_acc, val_loss, val_acc）
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # 学習
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 検証
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 履歴を保存
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 最良モデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved (val_acc: {val_acc:.4f})")
    
    return history


# ==============================================================================
# 評価
# ==============================================================================
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Tuple[float, Dict, np.ndarray, List[Tuple]]:
    """
    モデルを評価する
    
    Args:
        model: モデル
        dataloader: テストデータローダー
        device: デバイス
        class_names: クラス名のリスト
    
    Returns:
        (accuracy, classification_report_dict, confusion_matrix, misclassified_samples)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_images = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(inputs.cpu())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Classification Report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 誤分類サンプル
    misclassified_indices = np.where(all_preds != all_labels)[0]
    misclassified_samples = [
        (all_images[i], all_labels[i], all_preds[i])
        for i in misclassified_indices[:12]  # 最大12枚
    ]
    
    return accuracy, report, cm, misclassified_samples


def get_misclassified_samples_with_paths(
    model: nn.Module,
    data_dir: str,
    device: torch.device,
    class_names: List[str],
    max_samples: int = 12
) -> List[Tuple[str, int, int]]:
    """
    誤分類サンプルをパス付きで取得する
    
    Args:
        model: モデル
        data_dir: データディレクトリ
        device: デバイス
        class_names: クラス名のリスト
        max_samples: 最大サンプル数
    
    Returns:
        (画像パス, 正解ラベル, 予測ラベル)のリスト
    """
    model.eval()
    
    transform = get_val_transforms()
    test_dataset = datasets.ImageFolder(Path(data_dir) / "test", transform=transform)
    
    misclassified = []
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_dataset):
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            
            if pred.item() != label:
                path = test_dataset.imgs[idx][0]
                misclassified.append((path, label, pred.item()))
                
                if len(misclassified) >= max_samples:
                    break
    
    return misclassified


# ==============================================================================
# 可視化
# ==============================================================================
def plot_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    学習履歴をプロットする
    
    Args:
        history: 学習履歴
        save_path: 保存パス
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"History plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    混同行列をプロットする
    
    Args:
        cm: 混同行列
        class_names: クラス名のリスト
        save_path: 保存パス
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 数値を表示
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_misclassified_examples(
    misclassified: List[Tuple[str, int, int]],
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    誤分類例をプロットする
    
    Args:
        misclassified: (画像パス, 正解ラベル, 予測ラベル)のリスト
        class_names: クラス名のリスト
        save_path: 保存パス
    """
    n = len(misclassified)
    if n == 0:
        print("No misclassified examples found!")
        return
    
    # グリッドサイズを計算
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (path, true_label, pred_label) in enumerate(misclassified):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # 画像を読み込んで表示
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontsize=10
        )
        ax.axis("off")
    
    # 余ったセルを非表示
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis("off")
    
    plt.suptitle("Misclassified Examples", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Misclassified examples saved to {save_path}")
    
    plt.show()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    正規化されたテンソルを元に戻す
    
    Args:
        tensor: 正規化されたテンソル (C, H, W)
    
    Returns:
        numpy配列 (H, W, C)
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    return img


# ==============================================================================
# 結果保存
# ==============================================================================
def save_metrics(
    accuracy: float,
    report: Dict,
    save_path: str,
    config: Optional[Dict] = None
) -> None:
    """
    評価メトリクスをJSONファイルに保存する
    
    Args:
        accuracy: 精度
        report: classification_reportの辞書
        save_path: 保存パス
        config: 設定情報（任意）
    """
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "timestamp": datetime.now().isoformat(),
    }
    
    if config:
        metrics["config"] = config
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Metrics saved to {save_path}")
