from ui import run_webcam_mode, run_image_mode
from feature_matching import load_templates
import cv2

def main():
    print("Choose input mode:")
    print("1. Webcam")
    print("2. Image from file")
    choice = input("Enter 1 or 2: ")

    sift = cv2.SIFT_create()
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