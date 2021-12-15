import os
import time
from scipy import signal as sg
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable as cm

# img = cv.imread('/Users/admin/Documents/GitHub/IlluminationNormalization/Ouput B/ADC/iPhone11Pro_IMG_0316_575px_ADC.png', 0)
# ref = cv.imread('/Users/admin/Documents/GitHub/IlluminationNormalization/Groundtruth B/iPhone11Pro_IMG_0316_575px_GT.jpeg', 0)

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


# ab_pcqi_map = pcqi(a, b, gaussiankernel, 256)
# mpcqi = np.mean(ab_pcqi_map)
# print(mpcqi)
#
# # cv.imwrite('ab.png', ab_pcqi_map)
# plt.imshow(ab_pcqi_map, cmap='gray')
# plt.colorbar()
# plt.show()
ref = '/Users/admin/Documents/GitHub/IlluminationNormalization/Groundtruth A'

# chars = 'ADC/'
# chare = '_ADC'
# ipath = '/Users/admin/Documents/GitHub/IlluminationNormalization/Ouput B/ADC'
# opath = '/Users/admin/Documents/GitHub/IlluminationNormalization/ADCvsGT'
#
# path = ipath
#
# for each in glob.iglob(path + '/*.png'):
#     imstring = each
#     imname = imstring[imstring.find(chars) + 4: imstring.find(chare)]
#
#     he2 = cv.imread('Ouput B/HE2/' + imname + '_HE2.png', 0)
#     adc = cv.imread('Ouput B/ADC/' + imname + '_ADC.png', 0)
#     img = cv.imread(imstring, 0)
#     n, m = img.shape
#     dms = m, n
#     grd = cv.imread(ref + '/' + imname + '_GT.jpeg', 0)
#     grd = cv.resize(grd, dms, interpolation=cv.INTER_AREA)
#
#     pcqi_map = pcqi(img, grd, gaussiankernel, 256)
#     mpcqi = np.mean(pcqi_map)
#     print(imname + ': %d' % mpcqi)
#
#     # plt.imshow(pcqi_map, cmap='gray')
#     # plt.suptitle(imname)
#     # plt.colorbar()
#     # plt.show()
#
#     fig, (ax) = plt.subplots(1, 1, constrained_layout=True)
#     ax = plt.imshow(pcqi_map, cmap='gray')
#     plt.colorbar()
#     fig.suptitle(imname)
#     fig.savefig(os.path.join(opath, imname + '_pcqi_ADCvsGT.png'))

chars = 'A/'
chare = '_Mertens07'
# ipath = '/Users/admin/Documents/GitHub/IlluminationNormalization/Ouput B/HE1'
opath = '/Users/admin/Documents/GitHub/IlluminationNormalization/PLCEvsGT'

path = ref

for each in glob.iglob(path + '/*.png'):
    imstring = each
    imname = imstring[imstring.find(chars) + 2: imstring.find(chare)]

    adc = cv.imread('Ouput A/ADC/' + imname + '_ADC.png', 0)
    he1 = cv.imread('Ouput A/HE1/' + imname + '_HE1.png', 0)
    he2 = cv.imread('Ouput A/HE2/' + imname + '_HE2.png', 0)
    plce = cv.imread('Ouput A/PLCE/' + imname + '_PLCE.png', 0)
    # plce2 = cv.imread('Ouput B/PLCE2 and extensions/PLCE2/' + imname + '_PLCE2.png', 0)
    # plce2bce = cv.imread('Ouput B/PLCE2 and extensions/PLCE2xBCE/' + imname + '_PLCE2xBCE.png', 0)
    # retnet = cv.imread('Ouput B/RetinexNet/' + imname + '.jpeg', 0)

    img = plce

    n, m = img.shape
    dms = m, n
    grd = cv.imread(imstring, 0)
    grd = cv.resize(grd, dms, interpolation=cv.INTER_AREA)

    pcqi_map = pcqi(img, grd, gaussiankernel, 256)
    mpcqi = np.mean(pcqi_map)
    print(imname + ': %d' % mpcqi)

    # plt.imshow(pcqi_map, cmap='gray')
    # plt.suptitle(imname)
    # plt.colorbar()
    # plt.show()

    fig, (ax) = plt.subplots(1, 1, constrained_layout=True)
    ax = plt.imshow(pcqi_map, cmap='gray')
    plt.colorbar()
    fig.suptitle(imname)
    fig.savefig(os.path.join(opath, imname + '_PLCEvsGT.png'))