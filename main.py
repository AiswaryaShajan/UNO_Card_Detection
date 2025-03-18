import cv2 as cv
import numpy as np

#Callback for the trackbar
def nothing(x): 
    pass

# Making a window for the trackbars
cv.namedWindow("Trackbars")

# Create trackbars for HSV ranges
cv.createTrackbar("Lower-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("Lower-S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("Lower-V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("Upper-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("Upper-S", "Trackbars",0, 255, nothing)
cv.createTrackbar("Upper-V", "Trackbars", 0, 255, nothing)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get the real-time positions of trackbars
    lower_h = cv.getTrackbarPos("Lower-H", "Trackbars")
    lower_s = cv.getTrackbarPos("Lower-S", "Trackbars")
    lower_v = cv.getTrackbarPos("Lower-V", "Trackbars")
    upper_h = cv.getTrackbarPos("Upper-H", "Trackbars")
    upper_s = cv.getTrackbarPos("Upper-S", "Trackbars")
    upper_v = cv.getTrackbarPos("Upper-V", "Trackbars")

    # Define lower and upper HSV bounds using the trackbar positions
    lower_bound_green = np.array([lower_h, lower_s, lower_v],np.uint8)
    upper_bound_green = np.array([upper_h, upper_s, upper_v],np.uint8)
    lower_bound_red = np.array([136,147,147],np.uint8)
    upper_bound_red = np.array([179,255,255],np.uint8)

    # Create a mask 
    green_mask = cv.inRange(hsv_frame, lower_bound_green, upper_bound_green)
    red_mask = cv.inRange(hsv_frame, lower_bound_red, upper_bound_red)

    # Apply the mask to the original frame
    green_result = cv.bitwise_and(frame, frame, mask=green_mask)
    red_result = cv.bitwise_and(frame,frame,mask=red_mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("red Mask", red_mask)
    cv.imshow("green mask", green_mask)
    cv.imshow("green Result", green_result)
    cv.imshow("Red result", red_result)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
