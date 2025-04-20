import cv2

# Load image
img = cv2.imread('assets/camera/bandung_coffee_shop.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur it
blur = cv2.GaussianBlur(img, (7, 7), 0)

# Draw a rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

cv2.imshow('Gray', gray)
cv2.imshow('Blur', blur)
cv2.imshow('Rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()