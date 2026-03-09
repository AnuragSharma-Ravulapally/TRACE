import cv2
import face_recognition
import pickle
import os
import numpy as np
from datetime import datetime
import csv
import tkinter as tk
from tkinter import messagebox, ttk

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "models", "face_encodings.pickle")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "Attendance.csv")

# Initialize at module level so all functions can access it
marked_today = set()

def mark_attendance(name):
    """Logs the person's name and time to a CSV file."""
    if name not in marked_today:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        date_string = now.strftime("%Y-%m-%d")
        
        file_exists = os.path.isfile(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Time', 'Date', 'Status'])
            writer.writerow([name, time_string, date_string, 'Present'])
            
        print(f"✅ {name} marked PRESENT at {time_string}")
        marked_today.add(name)

def run_attendance_system(camera_index):
    if not os.path.exists(ENCODINGS_PATH):
        print(f"❌ Error: {ENCODINGS_PATH} not found.")
        return

    print("⏳ Loading TRACE Face Database...")
    data = pickle.loads(open(ENCODINGS_PATH, "rb").read())
    
    # Initialize webcam with the user-selected camera index
    video_capture = cv2.VideoCapture(camera_index)
    
    print("="*40)
    print(f"🎥 TRACE ATTENDANCE SCANNER LIVE (Camera {camera_index})")
    print("   Look at the camera to check-in.")
    print("   Press [Q] to quit.")
    print("="*40)

    while True:
        ret, frame = video_capture.read()
        if not ret: 
            print("❌ Error: Could not read from webcam!")
            break

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
                mark_attendance(name)

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Unknown":
                color = (0, 0, 255) 
                status_text = "ACCESS DENIED"
            else:
                color = (0, 255, 0) 
                status_text = "PRESENT"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, status_text, (left, top - 10), font, 0.8, color, 1)

        cv2.imshow('TRACE Auto-Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("👋 Scanner Closed.")

# --- UI & CAMERA DETECTION ---
def get_available_cameras():
    """Tests the first 5 indexes to see which cameras are available."""
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW is faster on Windows
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()
    return available_cameras

def launch_ui():
    """Builds the Tkinter interface to select a camera."""
    root = tk.Tk()
    root.title("TRACE - Camera Setup")
    root.geometry("320x150")
    
    tk.Label(root, text="Select Camera For Attendance:", font=("Arial", 12)).pack(pady=10)
    
    cameras = get_available_cameras()
    if not cameras:
        messagebox.showerror("Error", "No cameras detected!")
        root.destroy()
        return

    cam_var = tk.StringVar(value=cameras[0])
    dropdown = ttk.Combobox(root, textvariable=cam_var, values=cameras, state="readonly")
    dropdown.pack(pady=5)

    def on_start():
        selected_cam_str = cam_var.get()
        # Extract the number from "Camera 0", "Camera 1", etc.
        cam_index = int(selected_cam_str.split()[1])
        root.destroy() # Close the UI menu
        run_attendance_system(cam_index) # Launch the main camera feed

    tk.Button(root, text="Start TRACE Scanner", command=on_start, bg="#0078D7", fg="white", font=("Arial", 10, "bold")).pack(pady=15)
    
    root.mainloop()

if __name__ == "__main__":
    launch_ui()