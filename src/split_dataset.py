"""
データセット分割スクリプト
data/raw/ から train/val/test にデータを分割する

分割比: train 70% / val 15% / test 15%
乱数シード: 42
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple


def get_class_mapping() -> Dict[str, str]:
    """
    rawフォルダ名からクラス名へのマッピングを返す
    """
    return {
        "daytime": "day",
        "nighttime": "night",
        "sunrise": "sunrise_sunset"
    }


def split_files(
    files: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    ファイルリストをtrain/val/testに分割する
    
    Args:
        files: ファイルパスのリスト
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        seed: 乱数シード
    
    Returns:
        (train_files, val_files, test_files)のタプル
    """
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)
    
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def copy_files(
    files: List[str],
    src_dir: Path,
    dst_dir: Path,
    class_name: str
) -> int:
    """
    ファイルをコピーする
    
    Args:
        files: ファイル名のリスト
        src_dir: コピー元ディレクトリ
        dst_dir: コピー先ディレクトリ
        class_name: クラス名（サブディレクトリ名）
    
    Returns:
        コピーしたファイル数
    """
    dst_class_dir = dst_dir / class_name
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for f in files:
        src_path = src_dir / f
        dst_path = dst_class_dir / f
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
    
    return copied


def split_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    force: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    データセットを分割する
    
    Args:
        raw_dir: 元データのディレクトリ
        output_dir: 出力先ディレクトリ
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        seed: 乱数シード
        force: 既存の分割を上書きするかどうか
    
    Returns:
        各分割・クラスごとのファイル数を示す辞書
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # 出力ディレクトリの確認
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = output_path / split
        if split_dir.exists() and not force:
            print(f"警告: {split_dir} は既に存在します。forceオプションで上書き可能です。")
            # 既存のデータ数をカウントして返す
            stats = {}
            for split in splits:
                stats[split] = {}
                split_dir = output_path / split
                if split_dir.exists():
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            stats[split][class_dir.name] = len(list(class_dir.glob("*")))
            return stats
    
    # 既存の分割ディレクトリを削除（forceの場合）
    if force:
        for split in splits:
            split_dir = output_path / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
    
    class_mapping = get_class_mapping()
    stats = {split: {} for split in splits}
    
    # 各クラスについて分割
    for raw_class, target_class in class_mapping.items():
        src_class_dir = raw_path / raw_class
        
        if not src_class_dir.exists():
            print(f"警告: {src_class_dir} が見つかりません")
            continue
        
        # 画像ファイルを取得
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        files = [
            f.name for f in src_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not files:
            print(f"警告: {src_class_dir} に画像ファイルがありません")
            continue
        
        print(f"クラス '{raw_class}' -> '{target_class}': {len(files)} 枚の画像")
        
        # 分割
        train_files, val_files, test_files = split_files(
            files, train_ratio, val_ratio, seed
        )
        
        # コピー
        for split, file_list in zip(splits, [train_files, val_files, test_files]):
            dst_dir = output_path / split
            copied = copy_files(file_list, src_class_dir, dst_dir, target_class)
            stats[split][target_class] = copied
            print(f"  {split}: {copied} 枚")
    
    return stats


def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    """
    分割結果の統計を表示する
    """
    print("\n" + "=" * 50)
    print("データセット分割結果")
    print("=" * 50)
    
    for split, classes in stats.items():
        total = sum(classes.values())
        print(f"\n{split}: {total} 枚")
        for cls, count in sorted(classes.items()):
            print(f"  - {cls}: {count} 枚")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="データセットを分割する")
    parser.add_argument("--raw-dir", default="data/raw", help="元データのディレクトリ")
    parser.add_argument("--output-dir", default="data", help="出力先ディレクトリ")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="訓練データの割合")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="検証データの割合")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--force", action="store_true", help="既存の分割を上書きする")
    
    args = parser.parse_args()
    
    stats = split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force
    )
    
    print_stats(stats)
