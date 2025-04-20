import cv2
import numpy as np
import time
import dlib
import argparse
from scipy.spatial import distance as dist
from imutils import face_utils
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Initialize face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
try:
    model_path = "shape_predictor_68_face_landmarks.dat"
    # If the model file doesn't exist, provide download instructions
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        print("Please download the model from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place it in the same directory as this script.")
        exit(1)
    landmark_predictor = dlib.shape_predictor(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Constants
EYE_AR_THRESH = 0.25      # Eye aspect ratio threshold for blink detection
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for eye closure
HEAD_POSE_THRESHOLD = 15  # Degrees threshold for head pose

# Initialize counters
COUNTER = 0
TOTAL = 0
blink_counter = 0
frame_counter = 0

# Global state variables
fatigue_level = 20
distraction_level = 15
eyes_closed = False
looking_away = False
using_phone = False  # This would require a more complex model to detect
alert_active = False
last_blink_time = time.time()

# Calculate eye aspect ratio
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Get head pose estimation
def get_head_pose(shape):
    image_points = np.array([
        (shape[30]),     # Nose tip
        (shape[8]),      # Chin
        (shape[36]),     # Left eye left corner
        (shape[45]),     # Right eye right corner
        (shape[48]),     # Left mouth corner
        (shape[54])      # Right mouth corner
    ], dtype="double")
    
    # 3D model points (simplified)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # Camera matrix estimation (approximation)
    focal_length = 500
    size = (640, 480)
    center = (size[0] / 2, size[1] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    # Distortion coefficients
    dist_coeffs = np.zeros((4, 1))
    
    # Solve for pose
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles
    angles = np.degrees(np.array([
        np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]),  # pitch
        np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)),  # yaw
        np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])   # roll
    ]))
    
    return angles

# Function to update fatigue level based on blink rate
def update_fatigue_level():
    global fatigue_level, last_blink_time, blink_counter, frame_counter
    
    # Increase fatigue if eyes closed for a long time
    if eyes_closed:
        fatigue_level = min(100, fatigue_level + 0.5)
    else:
        # Decrease fatigue slowly when eyes are open
        fatigue_level = max(0, fatigue_level - 0.1)
    
    # Adjust fatigue based on blink rate (if abnormal)
    if frame_counter > 0:
        blink_rate = (blink_counter / frame_counter) * 30  # Normalized to 30fps
        
        # Normal blink rate is about 15-20 per minute
        # Too few or too many blinks indicate fatigue
        if blink_rate < 0.1 or blink_rate > 0.7:  
            fatigue_level = min(100, fatigue_level + 0.2)
    
    # Adjust based on time since last blink
    time_since_last_blink = time.time() - last_blink_time
    if time_since_last_blink > 5:  # Too long without blinking
        fatigue_level = min(100, fatigue_level + 0.3)

# Function to update distraction level based on head pose
def update_distraction_level(head_pose):
    global distraction_level, looking_away
    
    # Check if looking away based on head pose
    if abs(head_pose[1]) > HEAD_POSE_THRESHOLD or abs(head_pose[0]) > HEAD_POSE_THRESHOLD:
        looking_away = True
        distraction_level = min(100, distraction_level + 0.5)
    else:
        looking_away = False
        distraction_level = max(0, distraction_level - 0.2)

# Class for the Driver Monitoring application
class DriverMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Safety Monitoring System")
        self.root.geometry("1200x700")
        
        # Source selection frame
        self.source_frame = tk.Frame(root)
        self.source_frame.pack(pady=10)
        
        self.webcam_btn = tk.Button(self.source_frame, text="Use Webcam", command=self.use_webcam)
        self.webcam_btn.pack(side=tk.LEFT, padx=10)
        
        self.video_btn = tk.Button(self.source_frame, text="Load Video", command=self.load_video)
        self.video_btn.pack(side=tk.LEFT, padx=10)
        
        # Main display frame
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Video display frame
        self.video_display = tk.Label(self.display_frame)
        self.video_display.pack(side=tk.LEFT, padx=10)
        
        # Metrics frame
        self.metrics_frame = tk.Frame(self.display_frame, width=400)
        self.metrics_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH)
        
        # Fatigue level frame
        self.fatigue_frame = tk.Frame(self.metrics_frame, relief=tk.RAISED, bd=2)
        self.fatigue_frame.pack(pady=10, fill=tk.X)
        
        self.fatigue_label = tk.Label(self.fatigue_frame, text="Fatigue Level", font=("Arial", 14))
        self.fatigue_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.fatigue_value = tk.Label(self.fatigue_frame, text="0%", font=("Arial", 12))
        self.fatigue_value.pack(anchor=tk.W, padx=10)
        
        self.fatigue_bar = tk.Canvas(self.fatigue_frame, height=20, bg="gray")
        self.fatigue_bar.pack(fill=tk.X, padx=10, pady=5)
        self.fatigue_indicator = self.fatigue_bar.create_rectangle(0, 0, 0, 20, fill="green")
        
        self.eye_status = tk.Label(self.fatigue_frame, text="Eyes Open", font=("Arial", 10))
        self.eye_status.pack(anchor=tk.W, padx=10, pady=5)
        
        # Distraction level frame
        self.distraction_frame = tk.Frame(self.metrics_frame, relief=tk.RAISED, bd=2)
        self.distraction_frame.pack(pady=10, fill=tk.X)
        
        self.distraction_label = tk.Label(self.distraction_frame, text="Distraction Level", font=("Arial", 14))
        self.distraction_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.distraction_value = tk.Label(self.distraction_frame, text="0%", font=("Arial", 12))
        self.distraction_value.pack(anchor=tk.W, padx=10)
        
        self.distraction_bar = tk.Canvas(self.distraction_frame, height=20, bg="gray")
        self.distraction_bar.pack(fill=tk.X, padx=10, pady=5)
        self.distraction_indicator = self.distraction_bar.create_rectangle(0, 0, 0, 20, fill="green")
        
        self.head_status = tk.Label(self.distraction_frame, text="Looking Forward", font=("Arial", 10))
        self.head_status.pack(anchor=tk.W, padx=10, pady=5)
        
        # Alert status frame
        self.alert_frame = tk.Frame(self.metrics_frame, relief=tk.RAISED, bd=2)
        self.alert_frame.pack(pady=10, fill=tk.X)
        
        self.alert_label = tk.Label(self.alert_frame, text="Status: NORMAL", font=("Arial", 16))
        self.alert_label.pack(padx=10, pady=10)
        
        self.alert_message = tk.Label(self.alert_frame, text="Driver alert and attentive", font=("Arial", 12))
        self.alert_message.pack(padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready to start", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Video source
        self.video_source = None
        self.cap = None
        self.running = False
        self.after_id = None
        
    def use_webcam(self):
        self.stop_video()
        self.video_source = 0  # Default camera
        self.start_video()
        
    def load_video(self):
        self.stop_video()
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if filepath:
            self.video_source = filepath
            self.start_video()
    
    def start_video(self):
        if self.video_source is not None:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                self.status_bar.config(text=f"Error: Could not open video source {self.video_source}")
                return
            
            # Reset counters
            global COUNTER, TOTAL, blink_counter, frame_counter
            COUNTER = 0
            TOTAL = 0
            blink_counter = 0
            frame_counter = 0
            
            self.running = True
            self.process_video()
            self.status_bar.config(text=f"Processing video from {self.video_source}")
    
    def stop_video(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status_bar.config(text="Video stopped")
    
    def update_ui(self):
        # Update fatigue indicators
        fatigue_percent = int(fatigue_level)
        self.fatigue_value.config(text=f"{fatigue_percent}%")
        self.fatigue_bar.coords(self.fatigue_indicator, 0, 0, fatigue_percent * 3, 20)
        
        if fatigue_percent > 70:
            self.fatigue_bar.itemconfig(self.fatigue_indicator, fill="red")
        elif fatigue_percent > 50:
            self.fatigue_bar.itemconfig(self.fatigue_indicator, fill="orange")
        else:
            self.fatigue_bar.itemconfig(self.fatigue_indicator, fill="green")
            
        self.eye_status.config(text="Eyes Closed" if eyes_closed else "Eyes Open")
        
        # Update distraction indicators
        distraction_percent = int(distraction_level)
        self.distraction_value.config(text=f"{distraction_percent}%")
        self.distraction_bar.coords(self.distraction_indicator, 0, 0, distraction_percent * 3, 20)
        
        if distraction_percent > 60:
            self.distraction_bar.itemconfig(self.distraction_indicator, fill="red")
        elif distraction_percent > 40:
            self.distraction_bar.itemconfig(self.distraction_indicator, fill="orange")
        else:
            self.distraction_bar.itemconfig(self.distraction_indicator, fill="green")
            
        self.head_status.config(text="Looking Away" if looking_away else "Looking Forward")
        
        # Update alert status
        if fatigue_level > 70 or distraction_level > 60:
            self.alert_frame.config(bg="red")
            self.alert_label.config(text="Status: CRITICAL ALERT", bg="red", fg="white")
            self.alert_message.config(text="Driver attention required immediately", bg="red", fg="white")
        elif fatigue_level > 50 or distraction_level > 40:
            self.alert_frame.config(bg="orange")
            self.alert_label.config(text="Status: WARNING", bg="orange")
            self.alert_message.config(text="Potential safety risk detected", bg="orange")
        elif fatigue_level > 30 or distraction_level > 30:
            self.alert_frame.config(bg="yellow")
            self.alert_label.config(text="Status: CAUTION", bg="yellow")
            self.alert_message.config(text="Early signs of fatigue detected", bg="yellow")
        else:
            self.alert_frame.config(bg="green")
            self.alert_label.config(text="Status: NORMAL", bg="green", fg="white")
            self.alert_message.config(text="Driver alert and attentive", bg="green", fg="white")
    
    def process_video(self):
        if not self.running:
            return
        
        global COUNTER, TOTAL, fatigue_level, distraction_level, eyes_closed, looking_away
        global blink_counter, frame_counter, last_blink_time
        
        ret, frame = self.cap.read()
        if not ret:
            # Restart video if it's a file that ended
            if self.video_source != 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_video()
                    return
            else:
                self.stop_video()
                return
        
        frame_counter += 1
        
        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector(gray, 0)
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
            
            # Get facial landmarks
            shape = landmark_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Draw face bounding box
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get left and right eye coordinates
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calculate eye aspect ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Visualize eye landmarks
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Check for eye closure
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    eyes_closed = True
            else:
                # If eyes were closed and now open, count as a blink
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    blink_counter += 1
                    last_blink_time = time.time()
                COUNTER = 0
                eyes_closed = False
            
            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            # Get and visualize head pose
            try:
                head_pose = get_head_pose(shape)
                
                # Draw head pose vector
                nose_tip = shape[30]
                pitch, yaw, roll = head_pose
                
                # Draw direction arrows based on head pose
                length = 30
                dx = length * np.sin(np.radians(yaw))
                dy = -length * np.sin(np.radians(pitch))
                
                cv2.arrowedLine(frame, 
                                tuple(nose_tip), 
                                (int(nose_tip[0] + dx), int(nose_tip[1] + dy)),
                                (255, 0, 0), 2)
                
                # Update distraction level based on head pose
                update_distraction_level(head_pose)
                
                # Display head pose values
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Head pose error: {e}")
        
        # Update fatigue level
        update_fatigue_level()
        
        # Display EAR value and blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {TOTAL}", (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display fatigue and distraction levels
        cv2.putText(frame, f"Fatigue: {int(fatigue_level)}%", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Distraction: {int(distraction_level)}%", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the video display
        self.video_display.img_tk = img_tk
        self.video_display.config(image=img_tk)
        
        # Update UI elements
        self.update_ui()
        
        # Schedule the next frame
        self.after_id = self.root.after(10, self.process_video)

def main():
    parser = argparse.ArgumentParser(description='Driver Safety Monitoring System')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = DriverMonitoringApp(root)
    
    # If video path provided via command line
    if args.video:
        app.video_source = args.video
        app.start_video()
    
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_video(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()