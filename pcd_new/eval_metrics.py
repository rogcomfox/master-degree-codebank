import numpy as np
import cv2 as cv

def calc_rmse(ori_img, res_img):
    ori_img = ori_img.astype(np.float32)
    res_img = res_img.astype(np.float32)
    diff = (ori_img - res_img) ** 2
    sum_diff = np.sum(diff)
    num_pixel = float(res_img.shape[0] * res_img.shape[1])
    error_res = sum_diff / num_pixel
    return round(np.sqrt(error_res), 2)

def calc_psnr(ori_img, res_img):
    rmse_result = calc_rmse(ori_img, res_img)
    psnr_res = 20 * np.log10(255./rmse_result)
    return round(psnr_res, 2)

def calc_ssim(ori_img, res_img):
    ori_img = ori_img.astype(np.float32)
    res_img = res_img.astype(np.float32)
    c1 = (0.01*255)**2 #0.01 = K1, 255 = max pixel image
    c2 = (0.03*255)**2 #0.03 = K2
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv.filter2D(ori_img, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(res_img, -1, window)[5:-5, 5:-5]
    mu1_mu2 = mu1 * mu2
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    sigma1_sq = cv.filter2D(ori_img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(res_img**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(ori_img * res_img, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def calc_me(ori_img, res_img): #me -> Missclassification Error
    a = ''
