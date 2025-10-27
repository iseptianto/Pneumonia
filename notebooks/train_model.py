#!/usr/bin/env python3
"""
Train Pneumonia Detection Model from Chest X-Ray Images
This script creates a new .h5 model file for deployment
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def download_dataset():
    """Download the chest X-ray pneumonia dataset"""
    try:
        import kagglehub
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def create_model():
    """Create improved CNN model with ResNet50"""
    print("🏗️ Building model...")

    # Load ResNet50 base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("✅ Model built successfully")
    return model

def create_data_generators(base_dir):
    """Create data generators for training"""
    print("📊 Creating data generators...")

    train_dir = os.path.join(base_dir, 'chest_xray', 'train')
    val_dir = os.path.join(base_dir, 'chest_xray', 'val')
    test_dir = os.path.join(base_dir, 'chest_xray', 'test')

    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print("✅ Data generators created")
    return train_generator, validation_generator, test_generator

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks"""
    print("🚀 Starting training...")

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))
    print(f"📊 Class weights: {class_weights}")

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    EPOCHS = 20

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    print("✅ Training completed")
    return history

def evaluate_model(model, test_generator):
    """Evaluate the model"""
    print("📊 Evaluating model...")

    # Evaluate on test set
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_generator, verbose=1)

    print(f"\\n📈 Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {2 * test_precision * test_recall / (test_precision + test_recall):.4f}")

    # Classification report
    y_true = test_generator.classes
    y_pred = (model.predict(test_generator) > 0.5).astype(int)

    print("\\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    return test_acc

def save_model(model, accuracy):
    """Save the trained model"""
    print("💾 Saving model...")

    # Save the model
    model_filename = 'pneumonia_resnet50_trained.h5'
    model.save(model_filename)
    print(f"✅ Model saved as '{model_filename}'")

    # Also save in Keras format
    model.save('pneumonia_resnet50_trained.keras')
    print("✅ Model also saved in Keras format")

    # Check file size
    file_size = os.path.getsize(model_filename) / (1024 * 1024)  # MB
    print(f"📁 Model file size: {file_size:.2f} MB")

    if file_size > 50:
        print("⚠️ Model file > 50MB - upload to Google Drive")
        upload_to_drive = True
    else:
        print("✅ Model file ≤ 50MB - can be stored locally")
        upload_to_drive = False

    return model_filename, upload_to_drive

def plot_training_history(history):
    """Plot training history"""
    print("📈 Plotting training history...")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training history plot saved as 'training_history.png'")

def main():
    """Main training pipeline"""
    print("🩺 Pneumonia Detection Model Training")
    print("=" * 50)

    # Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset. Exiting.")
        return

    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(dataset_path)

    # Create model
    model = create_model()

    # Train model
    history = train_model(model, train_gen, val_gen)

    # Evaluate model
    accuracy = evaluate_model(model, test_gen)

    # Save model
    model_file, needs_drive = save_model(model, accuracy)

    # Plot training history
    plot_training_history(history)

    print("\\n🎉 Training pipeline completed!")
    print(f"📁 Model saved: {model_file}")
    print(f"📊 Final accuracy: {accuracy:.4f}")
    if needs_drive:
        print("☁️ Upload model to Google Drive for deployment")
    else:
        print("💻 Model can be stored locally")

if __name__ == "__main__":
    main()