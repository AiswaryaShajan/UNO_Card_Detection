import cv2 as cv
import numpy as np
from card_detection import detect_card  # Import your warp function from separate file

def match_template_on_card(card_image, template_path):
    """Match the template on the warped card image only."""
    # Convert card to grayscale and binary
    gray_card = cv.cvtColor(card_image, cv.COLOR_BGR2GRAY)
    _, binary_card = cv.threshold(gray_card, 128, 255, cv.THRESH_BINARY)

    # Load template in grayscale
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        print("Error loading template.")
        return

    # Resize template to smaller size for corner matching
    template = cv.resize(template, (60, 60), interpolation=cv.INTER_AREA)

    # Template Matching
    result = cv.matchTemplate(binary_card, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)

    # Draw result on card
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv.rectangle(card_image, top_left, bottom_right, (0, 255, 0), 2)

    # Show result
    cv.imshow("Template Match on Card", card_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# === Main Driver ===
if __name__ == "__main__":
    image_path = "image_dataset/yellow_7.jpg"
    template_path = "processed_templates/binary_7_temp.jpg"  # Update this

    warped_card = detect_card(image_path)
    if warped_card is not None:
        match_template_on_card(warped_card, template_path)
    else:
        print("Card not detected.")

