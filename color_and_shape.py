import cv2
import numpy as np
import os

# Load card templates
def load_templates(template_path="binary_cards/"):
    templates = {}
    for filename in os.listdir(template_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            card_name = os.path.splitext(filename)[0]  # Strip file extension
            img = cv2.imread(os.path.join(template_path, filename), cv2.IMREAD_GRAYSCALE)
            templates[card_name] = img
    return templates

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
        # Warp the card
        ordered_corners = order_points(approx.reshape(4, 2))
        width, height = 200, 300
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        warped = cv2.warpPerspective(frame, transform_matrix, (width, height))
        return warped
    return None

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

# Detect and match features using ORB
def match_card(warped_card, orb, bf, templates):
    gray_card = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    kp_card, des_card = orb.detectAndCompute(gray_card, None)

    best_match = None
    max_matches = 0

    for card_name, template in templates.items():
        kp_template, des_template = orb.detectAndCompute(template, None)
        matches = bf.match(des_card, des_template)

        # Sort matches by distance (lower = better)
        matches = sorted(matches, key=lambda x: x.distance)

        # Count good matches
        good_matches = [m for m in matches if m.distance < 150]
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match = card_name

    return best_match

# Detect the dominant color of the card
def detect_color(card_image):
    """Detects the dominant color of the UNO card using HSV color masks."""
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        "Red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),  # Lower red
                (np.array([160, 100, 100]), np.array([180, 255, 255]))],  # Upper red
        "Blue": [(np.array([100, 150, 50]), np.array([140, 255, 255]))],
        "Green": [(np.array([40, 100, 50]), np.array([80, 255, 255]))],
        "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
    }

    detected_color = None
    max_pixels = 0

    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv[:, :, 0])  # Initialize mask
        for lower, upper in ranges:
            mask += cv2.inRange(hsv, lower, upper)

        color_pixels = cv2.countNonZero(mask)

        # Find the color with the highest pixel count
        if color_pixels > max_pixels:
            max_pixels = color_pixels
            detected_color = color

    return detected_color

# Draw the detected color label on the card
def draw_color_label(card_image, color):
    """Draws the detected color text on the card image."""
    if color:
        text = color
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 50)  # Adjust position
        font_scale = 0.8
        font_color = (0, 0, 0)  # Black text
        thickness = 2

        # Add white outline for visibility
        cv2.putText(card_image, text, position, font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(card_image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Main function
def main():
    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Load card templates
    templates = load_templates()

    # Start webcam
    cap = cv2.VideoCapture(0)  # Use webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and warp the card
        warped_card = detect_card(frame)

        if warped_card is not None:
            # Match with templates
            best_match = match_card(warped_card, orb, bf, templates)

            # Detect color
            detected_color = detect_color(warped_card)

            # Display results
            if best_match:
                cv2.putText(frame, f"Detected: {best_match}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if detected_color:
                draw_color_label(frame, detected_color)

            # Optionally display the warped card
            cv2.imshow("Warped Card", warped_card)

        cv2.imshow("Card Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()