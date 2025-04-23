import cv2
import time

# Load video or webcam
cap = cv2.VideoCapture(0)

# Warm up the camera
print("Warming up camera...")
time.sleep(2)  # Wait for 2 seconds to let camera adjust

# Ensure camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Read multiple frames to allow camera to adjust
for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        exit()
    time.sleep(0.1)

# Read frame for ROI selection
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame for ROI selection")
    cap.release()
    exit()

# Show frame and select ROI
cv2.namedWindow("Select Object to Track")
bbox = cv2.selectROI("Select Object to Track", frame, False)
cv2.destroyWindow("Select Object to Track")

# Initialize tracker
tracker = cv2.TrackerCSRT_create()  # Also try KCF, MIL, etc.
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost Tracking", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
