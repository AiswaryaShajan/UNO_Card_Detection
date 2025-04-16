import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# Load card templates
def load_templates(template_path="binary_cards/"):
    templates = {}
    for filename in os.listdir(template_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            card_name = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(template_path, filename), cv2.IMREAD_GRAYSCALE)
            templates[card_name] = img
    return templates

# Order points for warping
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

# Detect and warp the card
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

# Apply dilation to enhance features
def apply_dilation(image, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

# Match card using SIFT features
def match_card(warped_card, sift, bf, templates):
    gray_card = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    kp_card, des_card = sift.detectAndCompute(gray_card, None)

    best_match = None
    max_matches = 0

    for card_name, template in templates.items():
        kp_template, des_template = sift.detectAndCompute(template, None)
        matches = bf.knnMatch(des_card, des_template, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match = card_name

    return best_match

# Detect dominant color
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

# Draw color label
def draw_color_label(card_image, color):
    if color:
        text = color
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 50)
        font_scale = 0.8
        font_color = (0, 0, 0)
        thickness = 2
        cv2.putText(card_image, text, position, font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(card_image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Resize image while preserving aspect ratio
def resize_image(image, max_width=800, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:  # Only resize if the image is larger than the specified size
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# Core detection + display
def process_and_display(frame, sift, bf, templates):
    warped_card = detect_card(frame)

    if warped_card is not None:
        # Apply dilation to the warped card
        dilated_card = apply_dilation(warped_card)

        best_match = match_card(dilated_card, sift, bf, templates)
        detected_color = detect_color(dilated_card)

        if best_match:
            cv2.putText(frame, f"Detected: {best_match}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if detected_color:
            draw_color_label(frame, detected_color)

        # Display the warped and dilated card in separate windows
        cv2.imshow("Warped Card", warped_card)
        cv2.imshow("Dilated Card", dilated_card)

    # Show the original frame with detections
    cv2.imshow("Card Detection", frame)

# Webcam mode
def run_webcam_mode(sift, bf, templates):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_and_display(frame, sift, bf, templates)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Image file mode with file browser
def run_image_mode(sift, bf, templates):
    # Open file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the root tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        print("No file selected.")
        return

    # Load the selected image
    frame = cv2.imread(file_path)
    if frame is None:
        print("Error: Could not read the selected image.")
        return

    # Resize the image while maintaining aspect ratio
    frame = resize_image(frame)

    # Process and display the resized image
    process_and_display(frame, sift, bf, templates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main menu
def main():
    print("Choose input mode:")
    print("1. Webcam")
    print("2. Image from file")
    choice = input("Enter 1 or 2: ")

    # Initialize SIFT
    sift = cv2.SIFT_create()
    # Use BFMatcher with L2 norm for SIFT
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    templates = load_templates()

    if choice == "1":
        run_webcam_mode(sift, bf, templates)
    elif choice == "2":
        run_image_mode(sift, bf, templates)
    else:
        print("Invalid choice. Please restart and enter 1 or 2.")
        
if __name__ == "__main__":
    main()