import cv2 as cv
import numpy as np

def preprocess_template(template):
    """ Converts template to binary to focus only on the number. """
    _, binary_template = cv.threshold(template, 128, 255, cv.THRESH_BINARY_INV)
    return binary_template

def match_template(image, template):
    """ Performs template matching with improved filtering. """
    image = cv.resize(image, (800, 600))  # Resize for consistency
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    template = preprocess_template(template)  # Convert template to binary

    result = cv.matchTemplate(gray_image, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Get the best match coordinates
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    # Draw a rectangle around the detected number
    cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Show the result
    cv.imshow('Matched Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Load the image and template in grayscale
image_path = "C:/Users/HP/OneDrive/Desktop/UNO_Card_Detection/image_dataset/yellow_3.jpg"
template_path = "C:/Users/HP/OneDrive/Desktop/UNO_Card_Detection/binary_cards/3.jpg"

image = cv.imread(image_path)
template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

if image is not None and template is not None:
    match_template(image, template)
else:
    print("Error loading images.")