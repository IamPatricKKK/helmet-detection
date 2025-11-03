"""
Script inference/prediction cho model nhận dạng mũ bảo hiểm
Sử dụng model đã train để dự đoán ảnh mới
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Cấu hình
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["no_helmet", "with_helmet"]


def load_model(model_path=None):
    """Load model đã train"""
    
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Khong tim thay model tai: {model_path}")
        print(f"[INFO] Hay train model truoc bang cach chay: python train_model.py")
        return None
    
    print(f"[INFO] Dang load model tu: {model_path}")
    model = keras.models.load_model(model_path)
    print("[OK] Model da duoc load thanh cong!")
    
    return model


def preprocess_image(image_path):
    """Tiền xử lý ảnh cho model"""
    
    try:
        # Đọc ảnh
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"[ERROR] Khong the doc anh: {image_path}")
        print(f"[ERROR] {str(e)}")
        return None


def predict_image(model, image_path):
    """Dự đoán một ảnh"""
    
    # Preprocess
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[0]: predictions[0][0] * 100,
            CLASS_NAMES[1]: predictions[0][1] * 100
        }
    }


def predict_batch(model, image_dir):
    """Dự đoán nhiều ảnh trong thư mục"""
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Khong tim thay thu muc: {image_dir}")
        return []
    
    results = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                result = predict_image(model, file_path)
                if result:
                    result["filename"] = filename
                    results.append(result)
    
    return results


def visualize_prediction(image_path, prediction):
    """Hiển thị ảnh và kết quả dự đoán"""
    
    img = Image.open(image_path)
    
    plt.figure(figsize=(10, 5))
    
    # Hiển thị ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Hiển thị kết quả
    plt.subplot(1, 2, 2)
    classes = list(prediction["probabilities"].keys())
    probs = list(prediction["probabilities"].values())
    
    colors = ['red' if pred == prediction["class"] else 'gray' for pred in classes]
    bars = plt.barh(classes, probs, color=colors)
    plt.xlabel('Probability (%)')
    plt.title('Prediction Results')
    plt.xlim(0, 100)
    
    # Thêm giá trị trên mỗi bar
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        plt.text(prob + 2, i, f'{prob:.2f}%', va='center')
    
    # Thêm label
    predicted_text = f"Predicted: {prediction['class']}"
    confidence_text = f"Confidence: {prediction['confidence']:.2f}%"
    plt.text(0.5, -0.3, f"{predicted_text}\n{confidence_text}", 
             transform=plt.gca().transAxes, 
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def predict_from_camera(model):
    """Dự đoán từ camera real-time"""
    
    print("[INFO] Dang mo camera...")
    print("[INFO] Nhan 'q' de thoat")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Khong the mo camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        display_frame = cv2.resize(frame, (640, 480))
        
        # Preprocess cho model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
        img_array = np.expand_dims(frame_resized, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Hiển thị kết quả trên frame
        label = f"{predicted_class}: {confidence:.1f}%"
        color = (0, 255, 0) if predicted_class == "with_helmet" else (0, 0, 255)
        cv2.putText(display_frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Helmet Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Hàm main"""
    
    print("=" * 60)
    print("INFERENCE MODEL NHAN DIEN MU BAO HIEM")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Menu
    print("\n[MENU] Chon che do:")
    print("1. Predict mot anh")
    print("2. Predict nhieu anh tu thu muc")
    print("3. Real-time camera")
    print("4. Thoat")
    
    choice = input("\nNhap lua chon (1-4): ").strip()
    
    if choice == "1":
        # Predict một ảnh
        image_path = input("Nhap duong dan anh: ").strip()
        if os.path.exists(image_path):
            print("\n[INFO] Dang xu ly...")
            result = predict_image(model, image_path)
            if result:
                print(f"\n[RESULT] Class: {result['class']}")
                print(f"[RESULT] Confidence: {result['confidence']:.2f}%")
                print(f"[RESULT] Probabilities:")
                for class_name, prob in result["probabilities"].items():
                    print(f"  - {class_name}: {prob:.2f}%")
                
                # Visualize
                try:
                    visualize_prediction(image_path, result)
                except Exception as e:
                    print(f"[WARNING] Khong the hien thi anh: {str(e)}")
        else:
            print(f"[ERROR] Khong tim thay anh: {image_path}")
    
    elif choice == "2":
        # Predict nhiều ảnh
        image_dir = input("Nhap duong dan thu muc anh: ").strip()
        if os.path.exists(image_dir):
            print("\n[INFO] Dang xu ly...")
            results = predict_batch(model, image_dir)
            
            if results:
                print(f"\n[RESULTS] Da xu ly {len(results)} anh:")
                print("-" * 60)
                for result in results:
                    print(f"{result['filename']}: {result['class']} ({result['confidence']:.2f}%)")
            else:
                print("[WARNING] Khong tim thay anh nao trong thu muc")
        else:
            print(f"[ERROR] Khong tim thay thu muc: {image_dir}")
    
    elif choice == "3":
        # Real-time camera
        try:
            predict_from_camera(model)
        except Exception as e:
            print(f"[ERROR] Loi khi mo camera: {str(e)}")
    
    elif choice == "4":
        print("[INFO] Thoat chuong trinh")
    
    else:
        print("[ERROR] Lua chon khong hop le")


if __name__ == "__main__":
    main()

