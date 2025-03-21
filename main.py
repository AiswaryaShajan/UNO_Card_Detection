import cv2 as cv
import numpy as np

def process_image(image_path,scale_percent=30):
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

    # Draw contours on the original image
    result = image.copy()
    cv.drawContours(result, contours, -1, (0, 255, 0), 2)

    cv.imshow('Original Image', image)
    cv.imshow('gray', gray)
    cv.imshow('blur', blur)
    cv.imshow('edges', edges)
    cv.imshow('contours', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

process_image('image_dataset/red_2_shadow.jpeg')
