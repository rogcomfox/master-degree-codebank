import cv2 as cv
import os
import argparse
from utils_segment import kmeans_segment, post_processing, draw_edge, bgr_to_lab, bgr_to_hsv, bgr_to_hls, bgr_to_hsi, watershed_segment

#main implementation
def main_segment():
    img = cv.imread('main_image.png')
    img = cv.resize(img, (800,600))
    #color conversion to lab
    lab_img, lab_to_zero = bgr_to_lab(img)
    if os.path.exists('lab_img.jpg') == False and os.path.exists('lab_zero.jpg') == False:
        cv.imwrite('lab_img.jpg', lab_img)
        cv.imwrite('lab_zero.jpg', lab_to_zero)
    # because better using l=0, continue segmentation
    lab_segment = kmeans_segment(lab_to_zero, k=3)
    opening_lab, canny_lab = post_processing(lab_segment)
    img_edge = draw_edge(img, canny_lab)
    cv.imshow('apply_edge', img_edge)
    # cv.imshow('lab_segment',lab_segment)
    # cv.imshow('morph_result', opening_lab)
    # cv.imshow('canny_result', canny_lab)
    cv.waitKey(0)
    cv.destroyAllWindows()

# we compare watershed vs kmeans for segmentation task
def segment_comparison():
    img = cv.imread('main_image.png')
    img = cv.resize(img, (800,600))
    #conversion to lab
    _, lab_to_zero = bgr_to_lab(img)
    watershed_result = watershed_segment(lab_to_zero)
    cv.waitKey(0)
    cv.destroyAllWindows()

# we compare with several famous color space using kmean segmentation
def color_space_effect():
    img = cv.imread('main_image.png')
    img = cv.resize(img, (800,600))
    #lab color space (l=0)
    _, lab_to_zero = bgr_to_lab(img)
    lab_segment = kmeans_segment(lab_to_zero, k=3)
    opening_lab, canny_lab = post_processing(lab_segment)
    #hsv color space (h=0)
    _, hsv_to_zero = bgr_to_hsv(img)
    hsv_segment = kmeans_segment(hsv_to_zero)
    opening_hsv, canny_hsv = post_processing(hsv_segment)
    #hsl color space (h=0)
    _, hls_to_zero = bgr_to_hls(img)
    hls_segment = kmeans_segment(hls_to_zero)
    opening_hls, canny_hls = post_processing(hls_segment)
    cv.imshow('morph_lab', opening_lab)
    cv.imshow('morph_hsv', opening_hsv)
    cv.imshow('morph_hls', opening_hls)
    cv.imshow('canny_lab', canny_lab)
    cv.imshow('canny_hsv', canny_hsv)
    cv.imshow('canny_hls', canny_hls)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main_segment()
    # segment_comparison()
    # color_space_effect()