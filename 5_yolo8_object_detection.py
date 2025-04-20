import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # You can also use yolov8m.pt or yolov8l.pt for more power

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)

    # Render results: boxes, labels, and confidence scores
    frame = results.render()[0]  # Renders the bounding boxes and labels

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
