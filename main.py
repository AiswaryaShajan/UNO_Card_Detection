import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cv.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # Finding the HSV value from the webcam frame
    hsv_value = hsv_frame[center_y, center_x]
    print(f"HSV Value at center: {hsv_value}")


    # Define HSV values
    # lower_bound_red = np.array([ , , ,],np.uint8)
    # upper_bound_red = np.array([ , , ,],np.uint8)

    # # Create the mask 
    # red_mask1 = cv.inRange(hsv_frame, lower_bound_red, upper_bound_red)
    
   
    # Combining the masks using bitwise or
    # red_mask = cv.bitwise_or(red_mask1,red_mask2)

    # Apply the mask to the original frame
    # red_result = cv.bitwise_and(frame,frame,mask=red_mask)




    cv.imshow("Original Frame", frame)
    # cv.imshow("red Mask1", red_mask1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
