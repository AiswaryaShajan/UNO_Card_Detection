import cv2
import numpy as np

def detect_color(card_image):
    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "Red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))],
        "Blue": [(np.array([100, 150, 50]), np.array([140, 255, 255]))],
        "Green": [(np.array([40, 100, 50]), np.array([80, 255, 255]))],
        "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
    }

    detected_color = None
    max_pixels = 0

    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv[:, :, 0])
        for lower, upper in ranges:
            mask += cv2.inRange(hsv, lower, upper)

        color_pixels = cv2.countNonZero(mask)
        if color_pixels > max_pixels:
            max_pixels = color_pixels
            detected_color = color

    return detected_color