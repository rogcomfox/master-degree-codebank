# make image to negative
import cv2 as cv

img = cv.imread("sample_1.jpg")
img_not = cv.bitwise_not(img)
cv.imwrite("sample_invert.jpg",img_not)