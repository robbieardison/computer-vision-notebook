import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # You can also use yolov8m.pt or yolov8l.pt for more power

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render results: boxes, labels, and confidence scores
    for result in results:  # Iterate through the results
        frame = result.plot()  # Use plot() to render the bounding boxes and labels

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
