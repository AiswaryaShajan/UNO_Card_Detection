import cv2 as cv
import numpy as np

img=cv.imread('templates/1_temp.jpg')
gray_1_temp  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray_1_temp)
cv.imwrite('processed_templates/1_temp.jpg', gray_1_temp)
cv.waitKey(0)
cv.destroyAllWindows
