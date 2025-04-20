import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Store the first frame as background reference
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1 = cv2.GaussianBlur(frame1, (21, 21), 0)

import time

last_update = time.time()

while True:
    ret, frame2 = cap.read()
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Only update background every 1 second
    if time.time() - last_update > 1:
        frame1 = gray
        last_update = time.time()

    delta_frame = cv2.absdiff(frame1, gray)
    thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame2, "Motion Detected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Video", frame2)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Delta", delta_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# This code captures video from the webcam and detects motion by comparing the current frame with a reference frame.