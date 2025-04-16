import cv2
import os

def load_templates(template_path="templates/"):
    templates = {}
    for filename in os.listdir(template_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            card_name = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(template_path, filename), cv2.IMREAD_GRAYSCALE)
            templates[card_name] = img
    return templates

def match_card(warped_card, sift, bf, templates):
    gray_card = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    kp_card, des_card = sift.detectAndCompute(gray_card, None)

    best_match = None
    max_matches = 0

    for card_name, template in templates.items():
        kp_template, des_template = sift.detectAndCompute(template, None)
        matches = bf.knnMatch(des_card, des_template, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > max_matches:  # Corrected comparison
            max_matches = len(good_matches)
            best_match = card_name

    return best_match