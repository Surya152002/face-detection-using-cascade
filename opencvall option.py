import cv2
import numpy as np

# Load and display an image
image = cv2.imread('image.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform Canny edge detection on the image
edges = cv2.Canny(image, 100, 200)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize the image
resized_image = cv2.resize(image, (300, 300))
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotate the image
rows, cols = image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image translation
translation_matrix = np.float32([[1, 0, 50], [0, 1, -30]])  # Translate x by 50 and y by -30
translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
cv2.imshow('Translated Image', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image flipping
flipped_image = cv2.flip(image, 1)  # Flip horizontally
cv2.imshow('Flipped Image', flipped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply image masking
mask = np.zeros_like(gray_image)
mask[100:300, 200:400] = 255  # Define a rectangular mask
masked_image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to a different color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed image
cv2.imwrite('processed_image.jpg', hsv_image)
