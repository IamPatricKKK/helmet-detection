"""
Script to process and organize dataset for helmet detection model
Convert from data_collection to standard dataset structure for training
"""

import os
import sys
import shutil
import random
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter

# Adjust working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if os.getcwd() != project_root:
    os.chdir(project_root)
    sys.path.insert(0, project_root)

# Configuration
SOURCE_DIR = "data_collection"
DATASET_DIR = "dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure total ratio = 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.001, "Train/val/test ratios must sum to 1.0"

# Class directories
CLASSES = {
    "no_helmet": 0,  # No helmet
    "with_helmet": 1  # With helmet
}

# Accepted image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image(image_path):
    """Check if image is valid"""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify image is not corrupted
        return True
    except Exception as e:
        print(f"[WARNING] Invalid image: {image_path} - {str(e)}")
        return False


def get_image_files(folder_path):
    """Get list of all valid image files in directory"""
    image_files = []
    if not os.path.exists(folder_path):
        print(f"[WARNING] Directory does not exist: {folder_path}")
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
    """Create dataset directory structure"""
    print(f"[INFO] Creating directory structure for {DATASET_DIR}...")
    
    # Create root dataset directory
    if os.path.exists(DATASET_DIR):
        response = input(f"[WARNING] Directory {DATASET_DIR} already exists. Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(DATASET_DIR)
        else:
            print("[ERROR] Dataset creation cancelled")
            return False
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Create train/val/test structure for each class
    for split in ["train", "val", "test"]:
        for class_name in CLASSES.keys():
            split_dir = os.path.join(DATASET_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
    
    print(f"[OK] Directory structure created for {DATASET_DIR}")
    return True


def split_dataset(files, train_ratio, val_ratio, test_ratio):
    """Split dataset into train/val/test"""
    random.shuffle(files)
    total = len(files)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def copy_and_rename_images(source_folder, destination_folder, files, class_name, split_name):
    """Copy and rename images to destination directory"""
    copied_files = []
    
    for idx, filename in enumerate(files):
        source_path = os.path.join(source_folder, filename)
        
        # Create new filename: class_split_index.extension
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
            print(f"[ERROR] Error copying {source_path}: {str(e)}")
    
    return copied_files


def create_metadata(all_files_info):
    """Create metadata CSV file"""
    print("[INFO] Creating metadata file...")
    
    df = pd.DataFrame(all_files_info)
    
    # Save metadata for each split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        metadata_path = os.path.join(DATASET_DIR, f"{split}_metadata.csv")
        split_df.to_csv(metadata_path, index=False, encoding='utf-8')
        print(f"[OK] Created {metadata_path}: {len(split_df)} images")
    
    # Save total metadata
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    df.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"[OK] Created {metadata_path}: {len(df)} images total")
    
    # Statistics
    print("\n[STATS] Dataset statistics:")
    print("-" * 50)
    stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(stats)
    print("-" * 50)
    print(f"\nTotal images: {len(df)}")
    for class_name in CLASSES.keys():
        count = len(df[df["class"] == class_name])
        print(f"  - {class_name}: {count} images")


def print_dataset_summary():
    """Print dataset summary"""
    print("\n" + "=" * 60)
    print("[SUMMARY] DATASET SUMMARY")
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
                    print(f"  - {class_name}: {count} images")
    
    print("\n" + "=" * 60)


def main():
    """Main function"""
    print("[START] Starting dataset processing...")
    print("=" * 60)
    
    # Check source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return
    
    # Create dataset structure
    if not create_dataset_structure():
        return
    
    # Process each class
    all_files_info = []
    
    for class_name, class_label in CLASSES.items():
        print(f"\n[PROCESS] Processing class: {class_name} (label: {class_label})")
        
        source_folder = os.path.join(SOURCE_DIR, class_name)
        image_files = get_image_files(source_folder)
        
        if not image_files:
            print(f"[WARNING] No valid images found in {source_folder}")
            continue
        
        print(f"   Found {len(image_files)} valid images")
        
        # Split train/val/test
        train_files, val_files, test_files = split_dataset(
            image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        
        print(f"   Split dataset: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Copy and rename images for each split
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if files:
                dest_folder = os.path.join(DATASET_DIR, split_name, class_name)
                copied_info = copy_and_rename_images(
                    source_folder, dest_folder, files, class_name, split_name
                )
                all_files_info.extend(copied_info)
                print(f"   [OK] Copied {len(copied_info)} images to {split_name}/")
    
    # Create metadata
    if all_files_info:
        create_metadata(all_files_info)
        print_dataset_summary()
        print("\n[OK] Completed! Dataset has been created successfully!")
    else:
        print("\n[ERROR] No images were processed")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    main()

