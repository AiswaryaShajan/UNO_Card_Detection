import cv2
from tkinter import filedialog, Tk
from utils import resize_image
from card_detection import detect_card
from feature_matching import match_card
from colour_detection import detect_color
from utils import draw_color_label, apply_dilation

def process_and_display(frame, sift, bf, templates):
    warped_card = detect_card(frame)

    if warped_card is not None:
        dilated_card = apply_dilation(warped_card)
        best_match = match_card(dilated_card, sift, bf, templates)
        detected_color = detect_color(dilated_card)

        if detected_color and best_match:
            draw_color_label(frame, detected_color, best_match)

        cv2.imshow("Card Detection", frame)
        cv2.imshow("Warped", warped_card)
        cv2.imshow("Dilated", dilated_card)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_image_mode(sift, bf, templates):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    root.destroy()

    if not file_path:
        print("No file selected.")
        return

    frame = cv2.imread(file_path)
    if frame is None:
        print("Error: Could not read the selected image.")
        return

    frame = resize_image(frame)
    process_and_display(frame, sift, bf, templates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()