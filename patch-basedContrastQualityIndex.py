import os
import time
from scipy import signal as sg
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

a = cv.imread('Ouput B/HE2/iPhone11Pro_IMG_0326_575px_HE2.png', 0)
b = cv.imread('Ouput B/PLCE/iPhone11Pro_IMG_0326_575px_PLCE.png', 0)

gaussiankernel = np.array([[1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445],
                           [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                           [6.49510362, 25.90969361, 41.0435344, 25.90969361, 6.49510362],
                           [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                           [1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445]])


def pcqi(image1, image2, window, L):
    mu1 = sg.convolve2d(window, image1, mode='valid')
    mu2 = sg.convolve2d(window, image2, mode='valid')

    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)

    # sigma1_sq = filter2(window, img1. * img1, 'valid') - mu1_sq
    sigma1_sq = sg.convolve2d(window, np.multiply(image1, image1), mode='valid') - mu1_sq
    # sigma2_sq = filter2(window, img2. * img2, 'valid') - mu2_sq
    sigma2_sq = sg.convolve2d(window, np.multiply(image2, image2), mode='valid') - mu2_sq
    # sigma12 = filter2(window, img1. * img2, 'valid') - mu1_mu2
    sigma12 = sg.convolve2d(window, np.multiply(image1, image2), mode='valid') - mu1_mu2

    # sigma1_sq = max(0, sigma1_sq)
    sigma1_sq = np.argmax(sigma1_sq)
    if sigma1_sq < 0:
        sigma1_sq = 0

    # sigma2_sq = max(0, sigma2_sq)
    sigma2_sq = np.argmax(sigma2_sq)
    if sigma2_sq < 0:
        sigma2_sq = 0

    C = 3

    # pcqi_map = (4 / pi) .* atan((sigma12 + C)./ (sigma1_sq + C))
    pcqi_map = np.multiply((4 / np.pi), np.arctan(np.divide(sigma12 + C, sigma1_sq + C)))
    # pcqi_map = pcqi_map .* ((sigma12 + C)./ (sqrt(sigma1_sq).* sqrt(sigma2_sq) + C))
    pcqi_map = np.multiply(pcqi_map, np.divide(sigma12 + C, np.multiply(np.sqrt(sigma1_sq), np.sqrt(sigma2_sq)) + C))
    # pcqi_map = pcqi_map .* exp(-abs(mu1 - mu2) / L)
    pcqi_map = np.multiply(pcqi_map, np.exp(-np.abs(mu1 - mu2) / L))

    return pcqi_map


# mpcqi = np.mean(pcqi_map)

ab_pcqi_map = pcqi(a, b, gaussiankernel, 256)
mpcqi = np.mean(ab_pcqi_map)
print(mpcqi)

# cv.imwrite('ab.png', ab_pcqi_map)
plt.imshow(ab_pcqi_map, cmap='gray')
plt.colorbar()
plt.show()