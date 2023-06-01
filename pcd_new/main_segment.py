import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_segment import kmeans_segment, median_filter, rgb_to_lab

#main method
def main_segment():
    img = cv.imread('main_image.png')
    # normalize image
    img = cv.resize(img, (400,300))
    #color conversion to lab
    lab_img, b_channel = rgb_to_lab(img)
    #segmentation with k-means
    lab_segment = kmeans_segment(lab_img, 3)
    b_segment = kmeans_segment(b_channel, 3)
    b_median = median_filter(b_segment, size=9)
    lab_median = median_filter(lab_segment, size=9)
    cv.imshow('ori_img', img)
    cv.imshow('b_channel only', b_channel)
    cv.imshow('l_channel=0', lab_img)
    cv.imshow('b_channel_segment', b_segment)
    cv.imshow('lab_segment', lab_segment)
    cv.imshow('b_median', b_median)
    cv.imshow('lab_median', lab_median)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main_segment()