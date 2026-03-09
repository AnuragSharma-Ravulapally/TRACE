import cv2
import face_recognition
import pickle
import os
import numpy as np
from datetime import datetime
import csv

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "models", "face_encodings.pickle")
# We will save the attendance sheet in the main project folder
ATTENDANCE_FILE = os.path.join(BASE_DIR, "Attendance.csv")

def mark_attendance(name):
    """Logs the person's name and time to a CSV file."""
    # Check if we already marked them present in this exact session
    # (To prevent writing "Anurag" 30 times a second)
    if name not in marked_today:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        date_string = now.strftime("%Y-%m-%d")
        
        # Write to the CSV file
        file_exists = os.path.isfile(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Add headers if the file is brand new
            if not file_exists:
                writer.writerow(['Name', 'Time', 'Date', 'Status'])
            
            writer.writerow([name, time_string, date_string, 'Present'])
            
        print(f"✅ {name} marked PRESENT at {time_string}")
        marked_today.add(name)

def run_attendance_system():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"❌ Error: {ENCODINGS_PATH} not found.")
        return

    print("⏳ Loading TRACE Face Database...")
    data = pickle.loads(open(ENCODINGS_PATH, "rb").read())
    
    video_capture = cv2.VideoCapture(0)
    
    print("="*40)
    print("🎥 TRACE ATTENDANCE SCANNER LIVE")
    print("   Look at the camera to check-in.")
    print("   Press [Q] to quit.")
    print("="*40)

    # Global set to track who is already marked in this run
    global marked_today
    marked_today = set()

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # Shrink frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = data["names"][best_match_index]
                # Log attendance!
                mark_attendance(name)

            face_names.append(name)

        # Draw the graphics
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Unknown":
                color = (0, 0, 255) # Red
                status_text = "ACCESS DENIED"
            else:
                color = (0, 255, 0) # Green
                status_text = "PRESENT"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # Show Name
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # Show Status floating above the box
            cv2.putText(frame, status_text, (left, top - 10), font, 0.8, color, 1)

        cv2.imshow('TRACE Auto-Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("👋 Scanner Closed.")

if __name__ == "__main__":
    run_attendance_system()