import cv2
import easyocr

image = cv2.imread('../assets/ktp_sample_2.jpg')

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binarization
denoised = cv2.medianBlur(binary, 3)  # Noise removal

reader = easyocr.Reader(['id'], gpu=False)  # Use CPU

# Perform OCR
results = reader.readtext(denoised)

for detection in results:
    print("Text:", detection[1], "Confidence:", detection[2])

# Draw bounding boxes on the original image
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow('OCR Results', image)
cv2.waitKey(0)
cv2.destroyAllWindows()