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
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create a mask 
    mask = cv.inRange(hsv_frame, lower_bound, upper_bound)

    # Apply the mask to the original frame
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
