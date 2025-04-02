import cv2 as cv
from cv2.gapi import mask
import numpy as np

def order_points(pts):
    """ Orders the 4 corner points: TL, TR, BR, BL """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def detect_color(image):
    """ Detects the dominant color of the card """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Define color ranges (HSV format)
    color_ranges = {
        "Red": [(136,147,147), (179,255,255)],
        "Blue": [(100, 150, 0), (140,255,255)],
        "Green": [(40, 40, 40), (90, 255, 255)],#40, 40, 40), (90, 255, 255
        "Yellow": [(15,43,88), (30, 255, 255)]
    }
    
    detected_color = "Unknown"
    max_pixels = 0

    for color, (lower, upper) in color_ranges.items():
        mask = cv.inRange(hsv, np.array(lower), np.array(upper))
        pixel_count = cv.countNonZero(mask)
        
        if pixel_count > max_pixels:  
            max_pixels = pixel_count
            detected_color = color

    return detected_color

def warp_card(image, approx):
    """ Warps the detected card to a standard size and detects color """
    ordered_corners = order_points(approx.reshape(4, 2))
    
    width, height = 200, 300  # Standard card size
    
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    pre_warp = cv.getPerspectiveTransform(ordered_corners, dst_pts)
    warped = cv.warpPerspective(image, pre_warp, (width, height))

    # Detect Color
    card_color = detect_color(warped)
    print(f"Detected Card Color: {card_color}")
    
    return warped

def resize_with_aspect_ratio(image, target_width=600):
    """ Resizes image while maintaining aspect ratio """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    resized_image = cv.resize(image, (target_width, new_height), interpolation=cv.INTER_AREA)
    return resized_image

def process_image(image_path):
    """ Main function to process the image with aspect ratio resizing """
    # Read image 
    image = cv.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Resize while maintaining aspect ratio
    image = resize_with_aspect_ratio(image, target_width=600)

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Gaussian blur to remove noise 
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # Adaptive Thresholding for better contour detection
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # Canny Edge detection (used only if needed)
    edges = cv.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return

    # Filter out very small contours
    contours = [c for c in contours if cv.contourArea(c) > 1000]

    if not contours:
        print("No significant contours detected.")
        return

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Draw contours on the original image
    result = image.copy()
    cv.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
    epsilon = 0.02 * cv.arcLength(largest_contour, True)  # 2% of the arc length
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    # Check if it's a quadrilateral (4 corners)
    if len(approx) == 4:
        print("Found corners:", approx)
        for corner in approx:
            x, y = corner[0]  # Each corner is stored as [[x, y]]
            cv.circle(result, (x, y), 5, (255,0,0), -1)  # mark the corners

        # Warp the card and detect color
        warped_card = warp_card(image, approx)
        cv.imshow("Warped Card", warped_card)
    else:
        print("The detected contour is not a rectangle.")

    cv.imshow('Original Image', image)
    cv.imshow('Thresholded Image', thresh)
    cv.imshow('Edges', edges)
    cv.imshow('Contours', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Test with an image
process_image('image_dataset/yellow_9_1.jpg')  # Replace with your image path
