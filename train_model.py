import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
# Path to your generated GEI images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "TRACE_GEI_Ready")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "trace_gait_model.h5")
# Image size (Standard for GEI is often 64x64 or 128x128)
IMG_SIZE = (64, 64)

def load_data(data_dir):
    print("⏳ Loading images...")
    images = []
    labels = []
    
    files = os.listdir(data_dir)
    total_files = len(files)
    
    for i, f in enumerate(files):
        if not f.endswith('.png'): continue
        
        # Extract Subject ID from filename (e.g., "001_nm01_090.png")
        # We assume the first part before the underscore is the ID
        subject_id = f.split('_')[0]
        
        path = os.path.join(data_dir, f)
        img = cv2.imread(path, 0) # Read as grayscale
        
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(subject_id)
            
        if i % 500 == 0:
            print(f"Processed {i}/{total_files} images...")

    print(f"✅ Loaded {len(images)} images total.")
    return np.array(images), np.array(labels)

def build_model(num_classes):
    model = models.Sequential([
        # 1st Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D((2, 2)),
        
        # 2nd Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 3rd Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        # Dense Layers (The "Decision Making" part)
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax') # Output layer
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    X, y_raw = load_data(DATA_DIR)
    
    # 2. Preprocess Data
    # Normalize pixel values to be between 0 and 1
    X = X.astype('float32') / 255.0
    # Reshape for CNN (add channel dimension) -> (Num_Images, 64, 64, 1)
    X = X.reshape(X.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1)
    
    # Encode labels (Convert "001", "002" into 0, 1, 2...)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(np.unique(y))
    print(f"🔢 Detected {num_classes} unique people in the dataset.")
    
    # 3. Split into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build Model
    model = build_model(num_classes)
    model.summary()
    
    # 5. Train!
    print("🚀 Starting Training... (This might take a few minutes)")
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    
    # 6. Save
    model.save(MODEL_SAVE_PATH)
    print("="*40)
    print(f"🎉 Model Trained & Saved to: {MODEL_SAVE_PATH}")
    print(f"Final Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print("="*40)