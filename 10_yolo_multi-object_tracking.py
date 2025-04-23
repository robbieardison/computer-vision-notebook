import cv2
from ultralytics import YOLO
from sort.sort import Sort  # from the SORT repo
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Start webcam
cap = cv2.VideoCapture(0)

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
        if conf > 0.4:  # Confidence threshold
            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    # Update tracker with detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw tracked boxes
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Multi-Object Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()