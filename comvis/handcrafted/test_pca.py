from baseline_model_handcrafted import pca
import cv2
import matplotlib.pyplot as plt

img_test = cv2.imread('test_image.jpg')
img_test = cv2.resize(img_test, (224,224))
pca(img_test, 20)