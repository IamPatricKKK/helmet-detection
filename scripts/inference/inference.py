"""
Inference/prediction script for helmet detection model
Use trained model to predict new images
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Adjust working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if os.getcwd() != project_root:
    os.chdir(project_root)
    sys.path.insert(0, project_root)

# Configuration
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["no_helmet", "with_helmet"]


def load_model(model_path=None):
    """Load trained model"""
    
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        print(f"[INFO] Please train model first by running: python scripts/training/train_model.py")
        return None
    
    print(f"[INFO] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("[OK] Model loaded successfully!")
    
    return model


def preprocess_image(image_path):
    """Preprocess image for model"""
    
    try:
        # Load image
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"[ERROR] Cannot read image: {image_path}")
        print(f"[ERROR] {str(e)}")
        return None


def predict_image(model, image_path):
    """Predict a single image"""
    
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
    """Predict multiple images in directory"""
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directory not found: {image_dir}")
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
    """Display image and prediction results"""
    
    img = Image.open(image_path)
    
    plt.figure(figsize=(10, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display results
    plt.subplot(1, 2, 2)
    classes = list(prediction["probabilities"].keys())
    probs = list(prediction["probabilities"].values())
    
    colors = ['red' if pred == prediction["class"] else 'gray' for pred in classes]
    bars = plt.barh(classes, probs, color=colors)
    plt.xlabel('Probability (%)')
    plt.title('Prediction Results')
    plt.xlim(0, 100)
    
    # Add values on each bar
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        plt.text(prob + 2, i, f'{prob:.2f}%', va='center')
    
    # Add label
    predicted_text = f"Predicted: {prediction['class']}"
    confidence_text = f"Confidence: {prediction['confidence']:.2f}%"
    plt.text(0.5, -0.3, f"{predicted_text}\n{confidence_text}", 
             transform=plt.gca().transAxes, 
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def predict_from_camera(model):
    """Predict from camera in real-time"""
    
    print("[INFO] Opening camera...")
    print("[INFO] Press 'q' to exit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        display_frame = cv2.resize(frame, (640, 480))
        
        # Preprocess for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
        img_array = np.expand_dims(frame_resized, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Display results on frame
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
    """Main function"""
    
    print("=" * 60)
    print("INFERENCE MODEL - HELMET DETECTION")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Menu
    print("\n[MENU] Select mode:")
    print("1. Predict a single image")
    print("2. Predict multiple images from directory")
    print("3. Real-time camera")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Predict a single image
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            print("\n[INFO] Processing...")
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
                    print(f"[WARNING] Cannot display image: {str(e)}")
        else:
            print(f"[ERROR] Image not found: {image_path}")
    
    elif choice == "2":
        # Predict multiple images
        image_dir = input("Enter image directory path: ").strip()
        if os.path.exists(image_dir):
            print("\n[INFO] Processing...")
            results = predict_batch(model, image_dir)
            
            if results:
                print(f"\n[RESULTS] Processed {len(results)} images:")
                print("-" * 60)
                for result in results:
                    print(f"{result['filename']}: {result['class']} ({result['confidence']:.2f}%)")
            else:
                print("[WARNING] No images found in directory")
        else:
            print(f"[ERROR] Directory not found: {image_dir}")
    
    elif choice == "3":
        # Real-time camera
        try:
            predict_from_camera(model)
        except Exception as e:
            print(f"[ERROR] Error opening camera: {str(e)}")
    
    elif choice == "4":
        print("[INFO] Exiting program")
    
    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()

