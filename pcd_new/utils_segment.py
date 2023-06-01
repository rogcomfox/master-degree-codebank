import math
import cv2 as cv
import numpy as np

#color space lab
def bgr_to_lab(rgb_img):
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2LAB)
    #set l channel to 0
    lab_to_zero = lab_img.copy()
    lab_to_zero[:,:,0] = 0
    return lab_img, lab_to_zero

#color space hsv
def bgr_to_hsv(rgb_img):
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    hsv_to_zero = hsv_img.copy()
    hsv_to_zero[:,:,0] = 0
    return hsv_img, hsv_to_zero

#color space hls
def bgr_to_hls(rgb_img):
    hls_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2HLS)
    hls_to_zero = hls_img.copy()
    hls_to_zero[:,:,0] = 0
    return hls_img, hls_to_zero

#color space hsi
def bgr_to_hsi(rgb_img):
    with np.errstate(divide='ignore', invalid='ignore'):
        bgr = np.float32(rgb_img)/255
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #find intensity
        intensity = np.divide(blue + green + red, 3)
        # find saturation
        minimum = np.minimum(np.minimum(red, green), blue)
        saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)
        #find hue
        hue = np.copy(red)
        for i in range(0, blue.shape[0]):
            for j in range(0, blue.shape[1]):
                hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                            math.sqrt((red[i][j] - green[i][j])**2 + ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                hue[i][j] = math.acos(hue[i][j])

                if blue[i][j] <= green[i][j]:
                    hue[i][j] = hue[i][j]
                else:
                    hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
        #merge all
        hsi_img = cv.merge((hue, saturation, intensity))
        hsi_to_zero = hsi_img.copy()
        hsi_to_zero[:,:,0] = 0
        return hsi_img, hsi_to_zero

def watershed_segment(img):
    s = 0

def kmeans_segment(img, k=3):
    #reshape image
    img_pixel = img.reshape((-1, 3))
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
    res_img = segmented_data.reshape((img.shape))
    return res_img

def post_processing(img, median_size=5, morph_kernel=5, thresh_low=50, thresh_up=100):
    #first add median filter
    filter_img = cv.medianBlur(img, ksize=median_size)
    #then we using morphological operation (opening)
    kernel = np.ones((morph_kernel,morph_kernel), np.uint8)
    erosion_img = cv.erode(filter_img, kernel, iterations=2)
    dilation_img = cv.dilate(erosion_img, kernel, iterations=3)
    #then add canny edge operator
    canny_img = cv.Canny(dilation_img, thresh_low, thresh_up)
    return dilation_img, canny_img

# to remove background with edge detector
def draw_edge(ori_img, canny_mask):
    h, w = ori_img.shape[:2]
    #first find contour
    contours, _ = cv.findContours(canny_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_copy = ori_img.copy()
    result = cv.drawContours(img_copy, contours, -1, (0,255,0), 3)
    return result
