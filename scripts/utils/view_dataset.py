"""
Script to view dataset information and preview images
"""

import os
import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

# Adjust working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if os.getcwd() != project_root:
    os.chdir(project_root)
    sys.path.insert(0, project_root)

DATASET_DIR = "dataset"


def show_dataset_info():
    """Display dataset overview information"""
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Directory {DATASET_DIR} does not exist")
        return
    
    # Read metadata
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        
        print(f"\nTotal images: {len(df)}")
        print("\nDistribution by class:")
        class_counts = df["class"].value_counts()
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} images")
        
        print("\nDistribution by split:")
        split_counts = df["split"].value_counts()
        for split_name, count in split_counts.items():
            print(f"  - {split_name}: {count} images")
        
        print("\nDetails by split and class:")
        print("-" * 40)
        stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
        print(stats)
    else:
        print(f"[WARNING] metadata.csv file not found")


def preview_images(split="train", num_samples=6):
    """Preview some images from dataset"""
    print(f"\n[INFO] Previewing {num_samples} random images from {split}...")
    
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_dir):
        print(f"[ERROR] Directory not found: {split_dir}")
        return
    
    # Get images from each class
    images = []
    labels = []
    
    for class_name in ["no_helmet", "with_helmet"]:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if files:
                # Get random sample of images
                sample_files = random.sample(files, min(num_samples // 2, len(files)))
                for filename in sample_files:
                    images.append(os.path.join(class_dir, filename))
                    labels.append(class_name)
    
    # Display images
    if images:
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
        if num_samples == 2:
            axes = axes.reshape(2, 1)
        
        random.shuffle(list(zip(images, labels)))
        
        for idx, (img_path, label) in enumerate(zip(images[:num_samples], labels[:num_samples])):
            try:
                img = Image.open(img_path)
                row = idx // (num_samples // 2)
                col = idx % (num_samples // 2)
                
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_title(label, fontsize=10)
                ax.axis('off')
            except Exception as e:
                print(f"[WARNING] Cannot read image {img_path}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig("dataset_preview.png", dpi=150, bbox_inches='tight')
        print("[OK] Saved preview to dataset_preview.png")
        plt.show()
    else:
        print("[WARNING] No images found")


def main():
    """Main function"""
    show_dataset_info()
    
    try:
        import matplotlib.pyplot as plt
        preview_images(split="train", num_samples=6)
    except ImportError:
        print("\n[INFO] Matplotlib not installed. Skipping image preview.")
    except Exception as e:
        print(f"\n[WARNING] Cannot display preview: {str(e)}")


if __name__ == "__main__":
    random.seed(42)
    main()

