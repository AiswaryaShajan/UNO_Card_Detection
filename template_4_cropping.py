import cv2

# Load image
img = cv2.imread('templates/4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use normal threshold (white stays white)
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Try slightly wider crop of the top-left "4"
digit_crop = binary[18:75, 18:60]  # Adjusted box

# Save or display the result
cv2.imwrite('processed_templates/fixed_template_4.jpg', digit_crop)
cv2.rectangle(img, (18, 18), (60, 75), (0, 255, 0), 2)
cv2.imshow("Crop Area", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Template 4", digit_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
