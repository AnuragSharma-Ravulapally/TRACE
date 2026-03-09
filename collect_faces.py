import cv2
import os

# --- CONFIGURATION ---
# This gets the folder where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# It defines where to save the faces: TRACE_Project/dataset/Face_Dataset
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "Face_Dataset")

def create_face_dataset(name):
    # Create the user's specific folder (e.g., dataset/Face_Dataset/Anurag)
    path = os.path.join(DATASET_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 Created new folder: {path}")
    
    # Start Webcam (0 is usually the default laptop camera)
    cam = cv2.VideoCapture(0)
    
    print("="*40)
    print(f"📸 CAMERA READY FOR: {name.upper()}")
    print("--------------------------------")
    print("   [SPACE]  -> Take Photo")
    print("   [ESC]    -> Finish & Quit")
    print("="*40)
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: Could not read from webcam!")
            break
            
        # Show the video window
        cv2.imshow("TRACE Face Collector", frame)
        
        k = cv2.waitKey(1)
        
        if k % 256 == 27:
            # ESC pressed
            print("👋 Closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            save_path = os.path.join(path, img_name)
            cv2.imwrite(save_path, frame)
            print(f"✅ Saved: {img_name}")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ask for the name of the person being enrolled
    person_name = input("Enter the name of the person (e.g., Anurag): ")
    # Simple check to ensure name isn't empty
    if person_name.strip():
        create_face_dataset(person_name)
    else:
        print("❌ Name cannot be empty!")