"""
Training script for helmet detection model
Uses Transfer Learning with MobileNetV2
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Adjust working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if os.getcwd() != project_root:
    os.chdir(project_root)
    sys.path.insert(0, project_root)

# Configuration
DATASET_DIR = "dataset"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 2

# Labels
CLASS_NAMES = ["no_helmet", "with_helmet"]
CLASS_LABELS = {"no_helmet": 0, "with_helmet": 1}


def create_data_generators():
    """Create data generators with data augmentation for train and validation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation and test (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    # Test generator
    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, test_generator


def create_model():
    """Create model using Transfer Learning with MobileNetV2"""
    
    print("[INFO] Creating model with MobileNetV2...")
    
    # Load MobileNetV2 pre-trained weights (excluding top layer)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers (train only top layers first)
    base_model.trainable = False
    
    # Create new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[OK] Model created successfully!")
    return model


def train_model(model, train_generator, val_generator):
    """Train model"""
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Increase patience to 20 for longer training
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print("[INFO] Starting training...")
    print("=" * 60)
    
    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model (optional - comment if not needed)
    # model.save(os.path.join(MODEL_DIR, "final_model.h5"))
    print(f"\n[OK] Model saved at {MODEL_DIR}")
    
    return history


def plot_training_history(history):
    """Plot training history"""
    
    print("[INFO] Plotting training history...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
    print(f"[OK] Saved plot at {os.path.join(MODEL_DIR, 'training_history.png')}")
    # Uncomment the line below if you want to display the plot (requires GUI)
    # plt.show()


def evaluate_model(model, test_generator):
    """Evaluate model on test set"""
    
    print("\n[INFO] Evaluating model on test set...")
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\n[RESULTS] Test Loss: {test_loss:.4f}")
    print(f"[RESULTS] Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    print("\n[INFO] Classification Report:")
    print("-" * 60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print(f"[OK] Saved confusion matrix at {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")
    # Uncomment the line below if you want to display confusion matrix (requires GUI)
    # plt.show()
    
    return test_accuracy


def print_model_summary(model):
    """Print model summary"""
    print("\n" + "=" * 60)
    print("[MODEL SUMMARY]")
    print("=" * 60)
    model.summary()
    print("=" * 60)


def main():
    """Main function"""
    
    print("=" * 60)
    print("TRAINING MODEL - HELMET DETECTION")
    print("=" * 60)
    
    # Check dataset
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("[INFO] Please run prepare_dataset.py first!")
        return
    
    # Create data generators
    print("\n[STEP 1] Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators()
    
    print(f"[OK] Train: {train_generator.samples} images, {len(train_generator.class_indices)} classes")
    print(f"[OK] Val: {val_generator.samples} images")
    print(f"[OK] Test: {test_generator.samples} images")
    print(f"[OK] Classes: {train_generator.class_indices}")
    
    # Create model
    print("\n[STEP 2] Creating model...")
    model = create_model()
    print_model_summary(model)
    
    # Training
    print("\n[STEP 3] Starting training...")
    history = train_model(model, train_generator, val_generator)
    
    # Plot training history
    print("\n[STEP 4] Plotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n[STEP 5] Evaluating on test set...")
    test_accuracy = evaluate_model(model, test_generator)
    
    print("\n" + "=" * 60)
    print("[COMPLETED] Training completed!")
    print(f"[FINAL] Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"[FILES] Model files saved at directory: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()

