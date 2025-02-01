import cv2
import pytesseract
from pytesseract import Output

image = cv2.imread('../assets/ktp_sample_2.webp')

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binarization
denoised = cv2.medianBlur(binary, 3)  # Noise removal

# Perform OCR
custom_config = r'--oem 3 --psm 6 -l ind'  # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM)
text = pytesseract.image_to_string(denoised, config=custom_config)

print("Extracted Text:\n", text)

cv2.imshow('Preprocessed Image', denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()