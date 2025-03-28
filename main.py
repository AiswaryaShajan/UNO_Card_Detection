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

def process_image(image_path, scale_percent=30):
    # Read image 
    image = cv.imread(image_path)

    # Resize the image
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Gaussian blur to remove noise 
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # Canny Edge detection 
    edges = cv.Canny(blur, 30, 100)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    # Draw contours on the original image
    result = image.copy()
    cv.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
    epsilon = 0.02 * cv.arcLength(largest_contour, True)  # 2% of the arc length
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    # Check if it's a quadrilateral (4 corners)
    if len(approx) == 4:
        print("Found corners:", approx)
        for corner in approx:
            x, y = corner[0]  # Each corner is stored as [[x, y]]
            cv.circle(result, (x, y), 5, (255,0,0), -1)  # mark the corners

        # Warp the card
        warped_card = warp_card(image, approx)
        cv.imshow("warp", warped_card)
    else:
        print("The contour is not a rectangle.")

    cv.imshow('Original Image', image)
    # cv.imshow('gray', gray)
    # cv.imshow('blur', blur)
    # cv.imshow('edges', edges)
    # cv.imshow('contours', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

process_image('image_dataset/red_2_shadow.jpeg')
