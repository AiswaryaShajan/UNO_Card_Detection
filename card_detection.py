import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        ordered_corners = order_points(approx.reshape(4, 2))
        width, height = 200, 300
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        warped = cv2.warpPerspective(frame, transform_matrix, (width, height))
        return warped
    return None