"""
Main script to create dataset - allows users to choose dataset 1 or dataset 2
"""

import os
import sys

# Add current directory to path to import modules from the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def show_menu():
    """Display selection menu"""
    print("=" * 60)
    print("CREATE DATASET FOR HELMET DETECTION PROJECT")
    print("=" * 60)
    print("\n[1] Dataset (Simple standardization)")
    print("    - Copy and organize images")
    print("    - Fast and simple")
    print("    - Keep original images")
    print("\n[2] Dataset (OpenCV processing)")
    print("    - Face detection and crop")
    print("    - Image enhancement (CLAHE)")
    print("    - Standardize size to 224x224")
    print("    - Better quality but slower")
    print("    - Will overwrite 'dataset' folder if it exists")
    print("\n[3] Exit")
    print("-" * 60)


def run_prepare_dataset():
    """Run prepare_dataset.py"""
    print("\n[INFO] Running prepare_dataset.py...")
    try:
        import prepare_dataset
        prepare_dataset.main()
    except Exception as e:
        print(f"[ERROR] Error running prepare_dataset.py: {str(e)}")
        return False
    return True


def run_prepare_dataset_2():
    """Run prepare_dataset_2.py"""
    print("\n[INFO] Running prepare_dataset_2.py...")
    try:
        import prepare_dataset_2
        prepare_dataset_2.main()
    except Exception as e:
        print(f"[ERROR] Error running prepare_dataset_2.py: {str(e)}")
        return False
    return True


def show_dataset_info():
    """Display dataset information if it exists"""
    # Adjust path to find dataset at root
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset")
    if not os.path.exists(dataset_path):
        return
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    def count_images(dataset_path):
        count = {"train": 0, "val": 0, "test": 0}
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                for class_name in ["no_helmet", "with_helmet"]:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        count[split] += len([f for f in os.listdir(class_path) 
                                           if os.path.isfile(os.path.join(class_path, f))])
        return count
    
    ds_count = count_images(dataset_path)
    
    print(f"\n{'Split':<20} {'Number of images':<15}")
    print("-" * 60)
    
    for split in ["train", "val", "test"]:
        print(f"{split.capitalize():<20} {ds_count[split]:<15}")
    
    total = sum(ds_count.values())
    print(f"{'TOTAL':<20} {total:<15}")
    print("=" * 60)


def main():
    """Main function"""
    while True:
        show_menu()
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n[SELECTED] Create Dataset (simple standardization)")
            print("[NOTE] Will create/overwrite 'dataset' folder")
            if run_prepare_dataset():
                print("\n[OK] Dataset created successfully!")
                show_dataset_info()
            
            response = input("\nContinue? (y/n): ").strip().lower()
            if response != 'y':
                break
        
        elif choice == "2":
            print("\n[SELECTED] Create Dataset (OpenCV processing)")
            print("[NOTE] Will create/overwrite 'dataset' folder with OpenCV processing")
            if run_prepare_dataset_2():
                print("\n[OK] Dataset created successfully!")
                show_dataset_info()
            
            response = input("\nContinue? (y/n): ").strip().lower()
            if response != 'y':
                break
        
        elif choice == "3":
            print("\n[INFO] Exiting program")
            break
        
        else:
            print("\n[ERROR] Invalid choice! Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()

