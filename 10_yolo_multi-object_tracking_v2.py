import cv2
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker with better parameters
tracker = Sort(
    max_age=20,        # Maximum number of frames to keep alive a track without associated detections
    min_hits=3,        # Minimum number of associated detections before track is initialised
    iou_threshold=0.3  # Minimum IOU for match
)

# Start webcam
cap = cv2.VideoCapture(0)

# Get COCO class names
class_names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect with YOLO
    results = model(frame)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        
        # Lower confidence threshold and ensure valid class
        if conf > 0.3 and cls in class_names:  
            # Format: [x1, y1, x2, y2, confidence]
            detections.append([
                x1.item(), 
                y1.item(), 
                x2.item(), 
                y2.item(), 
                conf.item()
            ])
            
    # Update tracker only if we have detections
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
    else:
        tracked_objects = []

    # Draw tracked boxes and class names
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        
        # Different colors for different IDs
        color = ((obj_id * 50) % 255, (obj_id * 80) % 255, (obj_id * 120) % 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add ID and class name if available
        label = f'ID: {obj_id}'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame and FPS
    cv2.imshow("Multi-Object Tracking", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()