import cv2 as cv
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
        "Red": [(0, 120, 70), (10, 255, 255)],
        "Blue": [(100, 150, 0), (140, 255, 255)],
        "Green": [(40, 40, 40), (90, 255, 255)],
        "Yellow": [(20, 100, 100), (30, 255, 255)]
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
    cv.putText(warped, card_color, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return warped, card_color

def detect_card(frame):
    """ Detects a card in the given frame and warps it """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    
    # Adaptive thresholding for dynamic lighting conditions
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame, None  # No contours found, return original frame

    # Filter out very small contours
    contours = [c for c in contours if cv.contourArea(c) > 1000]

    if not contours:
        return frame, None  # No significant contours found

    largest_contour = max(contours, key=cv.contourArea)
    epsilon = 0.02 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        cv.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        for corner in approx:
            x, y = corner[0]
            cv.circle(frame, (x, y), 5, (255, 0, 0), -1)

        warped_card, detected_color = warp_card(frame, approx)
        cv.imshow("Warped Card", warped_card)

        return frame, detected_color

    return frame, None

def start_webcam():
    """ Opens the webcam and processes frames in real-time """
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detected_color = detect_card(frame)

        if detected_color:
            cv.putText(processed_frame, f"Color: {detected_color}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv.imshow('Webcam Feed', processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()
    cv.destroyAllWindows()

# Run the webcam function
start_webcam()
