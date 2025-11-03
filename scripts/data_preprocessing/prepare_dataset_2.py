"""
Dataset preprocessing script using OpenCV for image analysis and normalization
Improves dataset quality by:
- Detecting faces in images
- Cropping and normalizing resize
- Filtering high-quality images
- Processing images to focus on head/helmet region
"""

import os
import sys
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

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

# Ensure ratios sum to 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.001

# Class directories
CLASSES = {
    "no_helmet": 0,
    "with_helmet": 1
}

# Accepted image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Output size for normalization
OUTPUT_SIZE = (224, 224)

# Face detection parameters
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (50, 50)


def load_face_cascade():
    """Load Haar Cascade for face detection"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("[ERROR] Cannot load face cascade!")
            return None
        return face_cascade
    except Exception as e:
        print(f"[ERROR] Error loading face cascade: {str(e)}")
        return None


def detect_and_crop_face(image, face_cascade):
    """
    Detect face in image and crop face + overhead region (to include helmet)
    
    Returns:
        cropped_image: Cropped image or None if face not detected
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )
    
    if len(faces) == 0:
        return None
    
    # Get largest face (if multiple faces detected)
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Expand crop region to include helmet (add 30% height on top)
    expand_top = int(h * 0.3)
    expand_bottom = int(h * 0.1)
    expand_left = int(w * 0.1)
    expand_right = int(w * 0.1)
    
    # Calculate new coordinates
    x_new = max(0, x - expand_left)
    y_new = max(0, y - expand_top)
    w_new = min(image.shape[1] - x_new, w + expand_left + expand_right)
    h_new = min(image.shape[0] - y_new, h + expand_top + expand_bottom)
    
    # Crop image
    cropped = image[y_new:y_new+h_new, x_new:x_new+w_new]
    
    return cropped


def enhance_image(image):
    """
    Improve image quality:
    - Normalize brightness/contrast
    - Sharpening (optional)
    """
    # Convert to LAB color space for brightness adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def preprocess_image(image_path, face_cascade, enhance=True):
    """
    Process image: detect face, crop, resize, enhance
    
    Returns:
        processed_image: Processed image (BGR) or None
        has_face: True if face detected
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        # Detect and crop face
        cropped = detect_and_crop_face(image, face_cascade)
        
        if cropped is None:
            # If no face detected, use entire image
            cropped = image
            has_face = False
        else:
            has_face = True
        
        # Enhance image (optional)
        if enhance:
            cropped = enhance_image(cropped)
        
        # Resize to standard size
        resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        
        return resized, has_face
    
    except Exception as e:
        print(f"[WARNING] Error processing image {image_path}: {str(e)}")
        return None, False


def validate_image_quality(image_path):
    """
    Check basic image quality
    
    Returns:
        is_valid: True if image is valid
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Check minimum size
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return False
        
        # Check if image is corrupt
        if img.size == 0:
            return False
        
        return True
    except:
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
                if validate_image_quality(file_path):
                    image_files.append(filename)
    
    return image_files


def create_dataset_structure():
    """Create dataset directory structure"""
    print("[INFO] Creating dataset directory structure...")
    
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
    
    print("[OK] Dataset directory structure created")
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


def process_and_save_images(source_folder, destination_folder, files, class_name, split_name, face_cascade):
    """
    Process and save images to destination directory
    
    Returns:
        saved_files_info: List of saved file information
    """
    saved_files_info = []
    processed_count = 0
    face_detected_count = 0
    
    for idx, filename in enumerate(files):
        source_path = os.path.join(source_folder, filename)
        
        # Process image
        processed_image, has_face = preprocess_image(source_path, face_cascade, enhance=True)
        
        if processed_image is None:
            print(f"[SKIP] Cannot process: {filename}")
            continue
        
        # Create new filename
        ext = Path(filename).suffix
        new_filename = f"{class_name}_{split_name}_{idx:04d}{ext}"
        dest_path = os.path.join(destination_folder, new_filename)
        
        # Save image
        try:
            cv2.imwrite(dest_path, processed_image)
            saved_files_info.append({
                "original_name": filename,
                "new_name": new_filename,
                "path": dest_path,
                "class": class_name,
                "label": CLASSES[class_name],
                "split": split_name,
                "has_face": has_face
            })
            processed_count += 1
            if has_face:
                face_detected_count += 1
        except Exception as e:
            print(f"[ERROR] Error saving {dest_path}: {str(e)}")
    
    print(f"   [OK] Processed {processed_count}/{len(files)} images (face detected: {face_detected_count})")
    return saved_files_info


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
    
    # Save overall metadata
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    df.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"[OK] Created {metadata_path}: {len(df)} total images")
    
    # Statistics
    print("\n[STATS] Dataset statistics:")
    print("-" * 60)
    stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(stats)
    print("-" * 60)
    
    # Face detection statistics
    if "has_face" in df.columns:
        face_stats = df.groupby("has_face").size()
        print(f"\nFace Detection:")
        print(f"  - With face: {face_stats.get(True, 0)} images")
        print(f"  - Without face: {face_stats.get(False, 0)} images")
        print(f"  - Face detection rate: {face_stats.get(True, 0)/len(df)*100:.1f}%")
    
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
    print("=" * 60)
    print("CREATE DATASET WITH OPENCV IMAGE PROCESSING")
    print("=" * 60)
    
    # Check source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return
    
    # Load face cascade
    print("\n[INFO] Loading face detection model...")
    face_cascade = load_face_cascade()
    if face_cascade is None:
        print("[ERROR] Cannot load face cascade. Exiting program.")
        return
    print("[OK] Face cascade loaded successfully")
    
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
        
        # Process and save images for each split
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if files:
                dest_folder = os.path.join(DATASET_DIR, split_name, class_name)
                saved_info = process_and_save_images(
                    source_folder, dest_folder, files, class_name, split_name, face_cascade
                )
                all_files_info.extend(saved_info)
    
    # Create metadata
    if all_files_info:
        create_metadata(all_files_info)
        print_dataset_summary()
        print("\n[OK] Completed! Dataset has been created successfully!")
        print("\n[NOTE] Dataset (folder 'dataset') processed with OpenCV:")
        print("  - Face detection and crop")
        print("  - Image enhancement (CLAHE)")
        print("  - Resize to 224x224")
        print("  - Filter high quality images")
    else:
        print("\n[ERROR] No images were processed")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    main()

