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

def warp_card(image, approx):
    """ Warps the detected card to a standard size """
    ordered_corners = order_points(approx.reshape(4, 2))
    
    width, height = 200, 300  # Adjust as needed
    
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    pre_warp = cv.getPerspectiveTransform(ordered_corners, dst_pts)
    warped = cv.warpPerspective(image, pre_warp, (width, height))
    return warped

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    """ Resizes image while keeping the aspect ratio """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv.resize(image, dim, interpolation=inter)

def detect_card(image_path, scale_percent=50):
    # Read image 
    image = cv.imread(image_path)
    image = resize_with_aspect_ratio(image, width=600)  # Keep aspect ratio

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Gaussian blur to remove noise 
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # Canny Edge detection 
    edges = cv.Canny(blur, 30, 100)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return None

    largest_contour = max(contours, key=cv.contourArea)

    # Approximate contour to polygon
    epsilon = 0.02 * cv.arcLength(largest_contour, True)  
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    # Check if it's a quadrilateral (4 corners)
    if len(approx) == 4:
        print("Card detected!")
        warped_card = warp_card(image, approx)
        return warped_card
    else:
        print("The detected shape is not a rectangle.")
        return None

if __name__ == "__main__":
    card = detect_card('image_dataset/blue_7.jpg')
    if card is not None:
        cv.imshow("Warped Card", card)
        cv.waitKey(0)
        cv.destroyAllWindows()
import cv2 as cv
from tkinter import filedialog
from tkinter import Tk


def detect_card_from_image():
    print("🖼️ Image mode selected.")

    # Open file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", 1)  # Make sure the dialog stays on top
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()

    if not file_path:
        print("No image selected.")
        return

    card = detect_card(file_path)

    if card is not None:
        # Create a named window and set it to always stay on top
        cv.namedWindow("Warped Card", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Warped Card", cv.WND_PROP_TOPMOST, 1)  # Keeps the window on top
        cv.imshow("Warped Card", card)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Card detection failed.")


