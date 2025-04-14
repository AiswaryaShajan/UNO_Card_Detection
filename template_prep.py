import cv2 as cv
import numpy as np
import os

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

def detect_card(image_path):
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
        print(f"No contours found for {image_path}.")
        return None

    largest_contour = max(contours, key=cv.contourArea)

    # Approximate contour to polygon
    epsilon = 0.02 * cv.arcLength(largest_contour, True)  
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    # Check if it's a quadrilateral (4 corners)
    if len(approx) == 4:
        print(f"Card detected in {image_path}!")
        warped_card = warp_card(image, approx)
        return warped_card
    else:
        print(f"The detected shape in {image_path} is not a rectangle.")
        return None

def convert_to_binary(image):
    """ Converts an image to binary using thresholding """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale if not already
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)  # Apply binary thresholding
    return binary

def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            card = detect_card(file_path)
            
            if card is not None:
                binary_card = convert_to_binary(card)
                
                # Save binary image to the output folder
                output_path = os.path.join(output_folder, f"binary_{filename}")
                cv.imwrite(output_path, binary_card)
                print(f"Saved binary card to {output_path}")
            else:
                print(f"Skipping {filename}, card detection failed.")
        else:
            print(f"Skipping {filename}, not an image file.")

if __name__ == "__main__":
    input_folder = "image_dataset"  # Folder containing the input images
    output_folder = "binary_cards"  # Folder to save the binary cards

    process_folder(input_folder, output_folder)