import cv2 as cv
import numpy as np

def rgb_to_lab(rgb_img):
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2LAB)
    #set l channel to 0
    lab_img[:,:,0] = 0
    l, a, b = cv.split(lab_img)
    return lab_img, b

def kmeans_segment(lab_img, k=3):
    #reshape image
    img_pixel = lab_img.reshape((-1, 3))
    img_pixel = np.float32(img_pixel)

    # define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv.kmeans(
        img_pixel, 
        K=k, 
        bestLabels=None, 
        criteria=criteria, 
        attempts=10, 
        flags=cv.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    res_img = segmented_data.reshape((lab_img.shape))
    return res_img

def median_filter(lab_img, size=5):
    filter_img = cv.medianBlur(lab_img, ksize=size)
    return filter_img

# def post_process(res_img):
#     filter_img = cv.medianBlur(res_img, ksize=5)
#     th, im_th = cv.threshold(filter_img, 220, 255, cv.THRESH_BINARY_INV)
 
#     # Copy the thresholded image.
#     im_floodfill = im_th.copy()
    
#     # Mask used to flood filling.
#     # Notice the size needs to be 2 pixels than the image.
#     h, w = im_th.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
    
#     # Floodfill from point (0, 0)
#     cv.floodFill(im_floodfill, mask, (0,0), 255)
    
#     # Invert floodfilled image
#     im_floodfill_inv = cv.bitwise_not(im_floodfill)
    
#     # Combine the two images to get the foreground.
#     im_out = im_th | im_floodfill_inv

#     #add canny operator
#     edge_img = cv.Canny(im_out, threshold1=50, threshold2=150)
#     return edge_img
