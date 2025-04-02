import cv2 as cv
import numpy as np

def detect_color(card_image):
    """Detects the dominant color of the UNO card using HSV color masks."""
    
    # Convert to HSV color space
    hsv = cv.cvtColor(card_image, cv.COLOR_BGR2HSV)

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
            mask += cv.inRange(hsv, lower, upper)

        color_pixels = cv.countNonZero(mask)

        # Find the color with the highest pixel count
        if color_pixels > max_pixels:
            max_pixels = color_pixels
            detected_color = color

    return detected_color

def draw_color_label(card_image, color):
    """Draws the detected color text on the card image."""
    if color:
        text = color
        font = cv.FONT_HERSHEY_SIMPLEX
        position = (20, 50)  # Adjust position
        font_scale = 0.8
        font_color = (0,0,0)  # White text
        thickness = 2

        # Add white outline for visibility
        cv.putText(card_image, text, position, font, font_scale, (255,255,255), thickness + 2, cv.LINE_AA)
        cv.putText(card_image, text, position, font, font_scale, font_color, thickness, cv.LINE_AA)

if __name__ == "__main__":
    from card_detection import detect_card

    card = detect_card('image_dataset/red_2_shadow.jpeg')
    if card is not None:
        color = detect_color(card)
        draw_color_label(card, color)
        
        print(f"Detected Color: {color}")
        cv.imshow("Detected Card", card)
        cv.waitKey(0)
        cv.destroyAllWindows()
