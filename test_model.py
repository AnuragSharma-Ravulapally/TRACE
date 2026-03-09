import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "TRACE_GEI_Ready")
# Make sure this matches the name of the model you actually saved!
MODEL_PATH = os.path.join(BASE_DIR, "models", "trace_gait_model.h5") 

IMG_SIZE = (64, 64)

def predict_random_person():
    # 1. Load the trained brain
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    print("⏳ Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")

    # 2. Pick a random image from the dataset
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    random_file = random.choice(all_files)
    
    # The filename has the answer (e.g., "001_nm01_090.png")
    true_identity = random_file.split('_')[0]
    
    # 3. Prepare the image for the AI
    img_path = os.path.join(DATA_DIR, random_file)
    img = cv2.imread(img_path, 0)
    
    # Show the image to us (Humans)
    plt.imshow(img, cmap='gray')
    plt.title(f"Mystery Walker\n(True ID: {true_identity})")
    plt.axis('off')
    
    # Process for AI (Resize, Normalize, Reshape)
    img_ai = cv2.resize(img, IMG_SIZE)
    img_ai = img_ai.astype('float32') / 255.0
    img_ai = img_ai.reshape(1, 64, 64, 1) # Batch of 1
    
    # 4. Ask the AI
    print(f"\n🔍 Analyzing {random_file}...")
    predictions = model.predict(img_ai)
    
    # Get the top guess
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # Note: If you used LabelEncoder, class '0' might be person '001'.
    # For a quick check, we assume the classes are sorted. 
    # (To be 100% exact, we would need to load the encoder, but this is a good sanity check).
    
    print("="*40)
    print(f"🕵️‍♂️ TRUE IDENTITY:      Person {true_identity}")
    print(f"🤖 AI PREDICTION:      Class {predicted_class_index} (Subject {int(predicted_class_index)+1:03d}?)")
    print(f"📊 CONFIDENCE:         {confidence:.2f}%")
    print("="*40)
    
    plt.show()

if __name__ == "__main__":
    predict_random_person()