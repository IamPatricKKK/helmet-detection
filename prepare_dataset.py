"""
Script xử lý và tổ chức dataset cho model nhận dạng mũ bảo hiểm
Chuyển đổi từ data_collection sang cấu trúc dataset chuẩn cho training
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# Cấu hình
SOURCE_DIR = "data_collection"
DATASET_DIR = "dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Đảm bảo tổng tỷ lệ = 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.001, "Tỷ lệ train/val/test phải cộng lại bằng 1.0"

# Các thư mục class
CLASSES = {
    "no_helmet": 0,  # Không đội mũ
    "with_helmet": 1  # Có đội mũ
}

# Các định dạng ảnh được chấp nhận
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image(image_path):
    """Kiểm tra ảnh có hợp lệ không"""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify ảnh không bị corrupt
        return True
    except Exception as e:
        print(f"[WARNING] Anh khong hop le: {image_path} - {str(e)}")
        return False


def get_image_files(folder_path):
    """Lấy danh sách tất cả file ảnh hợp lệ trong thư mục"""
    image_files = []
    if not os.path.exists(folder_path):
        print(f"[WARNING] Thu muc khong ton tai: {folder_path}")
        return image_files
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = Path(filename).suffix.lower()
            if ext in VALID_EXTENSIONS:
                if validate_image(file_path):
                    image_files.append(filename)
    
    return image_files


def create_dataset_structure():
    """Tạo cấu trúc thư mục dataset"""
    print("[INFO] Dang tao cau truc thu muc dataset...")
    
    # Tạo thư mục gốc dataset
    if os.path.exists(DATASET_DIR):
        response = input(f"[WARNING] Thu muc {DATASET_DIR} da ton tai. Xoa va tao lai? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(DATASET_DIR)
        else:
            print("[ERROR] Huy tao dataset")
            return False
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Tạo cấu trúc train/val/test cho mỗi class
    for split in ["train", "val", "test"]:
        for class_name in CLASSES.keys():
            split_dir = os.path.join(DATASET_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
    
    print("[OK] Da tao cau truc thu muc dataset")
    return True


def split_dataset(files, train_ratio, val_ratio, test_ratio):
    """Chia dataset thành train/val/test"""
    random.shuffle(files)
    total = len(files)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def copy_and_rename_images(source_folder, destination_folder, files, class_name, split_name):
    """Copy và đổi tên ảnh vào thư mục đích"""
    copied_files = []
    
    for idx, filename in enumerate(files):
        source_path = os.path.join(source_folder, filename)
        
        # Tạo tên file mới: class_split_index.extension
        ext = Path(filename).suffix
        new_filename = f"{class_name}_{split_name}_{idx:04d}{ext}"
        dest_path = os.path.join(destination_folder, new_filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_files.append({
                "original_name": filename,
                "new_name": new_filename,
                "path": dest_path,
                "class": class_name,
                "label": CLASSES[class_name],
                "split": split_name
            })
        except Exception as e:
            print(f"[ERROR] Loi khi copy {source_path}: {str(e)}")
    
    return copied_files


def create_metadata(all_files_info):
    """Tạo file metadata CSV"""
    print("[INFO] Dang tao file metadata...")
    
    df = pd.DataFrame(all_files_info)
    
    # Lưu metadata cho từng split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        metadata_path = os.path.join(DATASET_DIR, f"{split}_metadata.csv")
        split_df.to_csv(metadata_path, index=False, encoding='utf-8')
        print(f"[OK] Da tao {metadata_path}: {len(split_df)} anh")
    
    # Lưu metadata tổng
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    df.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"[OK] Da tao {metadata_path}: {len(df)} anh tong cong")
    
    # Thống kê
    print("\n[STATS] Thong ke dataset:")
    print("-" * 50)
    stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(stats)
    print("-" * 50)
    print(f"\nTong so anh: {len(df)}")
    for class_name in CLASSES.keys():
        count = len(df[df["class"] == class_name])
        print(f"  - {class_name}: {count} anh")


def print_dataset_summary():
    """In tóm tắt dataset"""
    print("\n" + "=" * 60)
    print("[SUMMARY] TOM TAT DATASET")
    print("=" * 60)
    
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATASET_DIR, split)
        if os.path.exists(split_path):
            print(f"\n{split.upper()}:")
            for class_name in CLASSES.keys():
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                                if os.path.isfile(os.path.join(class_path, f))])
                    print(f"  - {class_name}: {count} anh")
    
    print("\n" + "=" * 60)


def main():
    """Hàm main"""
    print("[START] Bat dau xu ly dataset...")
    print("=" * 60)
    
    # Kiểm tra thư mục nguồn
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Khong tim thay thu muc {SOURCE_DIR}")
        return
    
    # Tạo cấu trúc dataset
    if not create_dataset_structure():
        return
    
    # Xử lý từng class
    all_files_info = []
    
    for class_name, class_label in CLASSES.items():
        print(f"\n[PROCESS] Dang xu ly class: {class_name} (label: {class_label})")
        
        source_folder = os.path.join(SOURCE_DIR, class_name)
        image_files = get_image_files(source_folder)
        
        if not image_files:
            print(f"[WARNING] Khong tim thay anh hop le trong {source_folder}")
            continue
        
        print(f"   Tim thay {len(image_files)} anh hop le")
        
        # Chia train/val/test
        train_files, val_files, test_files = split_dataset(
            image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        
        print(f"   Chia dataset: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Copy và đổi tên ảnh cho từng split
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if files:
                dest_folder = os.path.join(DATASET_DIR, split_name, class_name)
                copied_info = copy_and_rename_images(
                    source_folder, dest_folder, files, class_name, split_name
                )
                all_files_info.extend(copied_info)
                print(f"   [OK] Da copy {len(copied_info)} anh vao {split_name}/")
    
    # Tạo metadata
    if all_files_info:
        create_metadata(all_files_info)
        print_dataset_summary()
        print("\n[OK] Hoan thanh! Dataset da duoc tao thanh cong!")
    else:
        print("\n[ERROR] Khong co anh nao duoc xu ly")


if __name__ == "__main__":
    # Set seed cho reproducibility
    random.seed(42)
    main()

