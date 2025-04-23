import face_recognition
import cv2
import os
import numpy as np

# Load a sample image and learn how to recognize it
image_path = "assets/mantra.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

known_image = face_recognition.load_image_file(image_path)
known_face_locations = face_recognition.face_locations(known_image)
if not known_face_locations:
    raise ValueError("No faces found in the reference image")
known_encoding = face_recognition.face_encodings(known_image, known_face_locations)[0]

known_names = ["Lars"]
known_encodings = [known_encoding]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Ensure the frame is in RGB format
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        if face_locations:
            # Get face encodings for any faces in the picture
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, 
                face_locations,
                num_jitters=0  # Reduce jitters to speed up processing
            )

            # Loop through each face in this frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    name = known_names[matches.index(True)]

                # Scale back up face locations since we scaled our frame down
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
