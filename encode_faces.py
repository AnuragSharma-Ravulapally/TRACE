import face_recognition
import pickle
import os
import cv2

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "Face_Dataset")
ENCODINGS_PATH = os.path.join(BASE_DIR, "models", "face_encodings.pickle")

def encode_known_faces():
    print("🔍 Quantifying faces...")
    known_encodings = []
    known_names = []

    # Walk through the dataset folder
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Extract name from folder name (e.g., .../Anurag/image.jpg -> "Anurag")
                name = os.path.basename(root)
                image_path = os.path.join(root, file)
                
                # Load image
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces in the image
                boxes = face_recognition.face_locations(rgb_image, model="hog")
                
                # Compute the facial embedding (128-d vector)
                encodings = face_recognition.face_encodings(rgb_image, boxes)

                # Loop over the encodings (usually just one face per photo)
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)
    
    # Save the data
    print(f"✅ Serializing {len(known_encodings)} encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    
    with open(ENCODINGS_PATH, "wb") as f:
        f.write(pickle.dumps(data))
        
    print(f"🎉 Success! Face data saved to: {ENCODINGS_PATH}")

if __name__ == "__main__":
    encode_known_faces()