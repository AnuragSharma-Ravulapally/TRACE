import cv2
import face_recognition
import pickle
import os
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "models", "face_encodings.pickle")

def test_live_recognition():
    # 1. Load the known faces and embeddings
    if not os.path.exists(ENCODINGS_PATH):
        print(f"❌ Error: Could not find {ENCODINGS_PATH}")
        print("Did you run 'encode_faces.py' yet?")
        return

    print("⏳ Loading face brain...")
    data = pickle.loads(open(ENCODINGS_PATH, "rb").read())
    print(f"✅ Loaded encodings for: {list(set(data['names']))}")

    # 2. Start Webcam
    video_capture = cv2.VideoCapture(0)
    
    print("="*40)
    print("   [Q] -> Quit")
    print("="*40)

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('TRACE Face Verification', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_live_recognition()