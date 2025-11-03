"""
Script để xem thông tin và preview dataset
"""

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

DATASET_DIR = "dataset"


def show_dataset_info():
    """Hiển thị thông tin tổng quan về dataset"""
    print("=" * 60)
    print("THONG TIN DATASET")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Thu muc {DATASET_DIR} khong ton tai")
        return
    
    # Đọc metadata
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        
        print(f"\nTong so anh: {len(df)}")
        print("\nPhan bo theo class:")
        class_counts = df["class"].value_counts()
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} anh")
        
        print("\nPhan bo theo split:")
        split_counts = df["split"].value_counts()
        for split_name, count in split_counts.items():
            print(f"  - {split_name}: {count} anh")
        
        print("\nChi tiet theo split va class:")
        print("-" * 40)
        stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
        print(stats)
    else:
        print(f"[WARNING] Khong tim thay file metadata.csv")


def preview_images(split="train", num_samples=6):
    """Xem preview một số ảnh từ dataset"""
    print(f"\n[INFO] Dang xem {num_samples} anh ngau nhien tu {split}...")
    
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_dir):
        print(f"[ERROR] Khong tim thay thu muc {split_dir}")
        return
    
    # Lấy ảnh từ mỗi class
    images = []
    labels = []
    
    for class_name in ["no_helmet", "with_helmet"]:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if files:
                # Lấy ngẫu nhiên một số ảnh
                sample_files = random.sample(files, min(num_samples // 2, len(files)))
                for filename in sample_files:
                    images.append(os.path.join(class_dir, filename))
                    labels.append(class_name)
    
    # Hiển thị ảnh
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
                print(f"[WARNING] Khong the doc anh {img_path}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig("dataset_preview.png", dpi=150, bbox_inches='tight')
        print("[OK] Da luu preview vao dataset_preview.png")
        plt.show()
    else:
        print("[WARNING] Khong tim thay anh nao")


def main():
    """Hàm main"""
    show_dataset_info()
    
    try:
        import matplotlib.pyplot as plt
        preview_images(split="train", num_samples=6)
    except ImportError:
        print("\n[INFO] Matplotlib khong duoc cai dat. Bo qua preview anh.")
    except Exception as e:
        print(f"\n[WARNING] Khong the hien thi preview: {str(e)}")


if __name__ == "__main__":
    random.seed(42)
    main()

