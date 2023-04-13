from baseline_model_handcrafted import pca
import cv2
import matplotlib.pyplot as plt

img_test = cv2.imread('C://Users/ROG_is_Awesome/Downloads/vehicle_copy/test/testset/000071.jpg')
img_test = cv2.resize(img_test, (224,224))
img_compress_1 = pca(img_test, 20)
img_compress_2 = pca(img_test, 50)
img_compress_3 = pca(img_test, 100)
plt.subplot(1,3,1)
plt.title("PCA = 20")
plt.imshow(img_compress_1)
plt.subplot(1,3,2)
plt.title("PCA = 50")
plt.imshow(img_compress_2)
plt.subplot(1,3,3)
plt.title("PCA = 100")
plt.imshow(img_compress_3)
plt.savefig('pca_test.png', bbox_inches='tight')
plt.show()