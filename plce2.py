from scipy import signal as sg
import numpy as np
import cv2 as cv
import glob
import os
from numpy import diff
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

gaussiankernel = np.array([[1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445],
                           [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                           [6.49510362, 25.90969361, 41.0435344, 25.90969361, 6.49510362],
                           [4.10018648, 16.35610171, 25.90969361, 16.35610171, 4.10018648],
                           [1.0278445, 4.10018648, 6.49510362, 4.10018648, 1.0278445]])

dx = 0.01
tracking_list_beta = list(np.arange(1, 2.85, 0.01))
L = 256


def rescale(matrix):
    vmin = matrix.min()
    vmax = matrix.max()
    matrix = 255 * (matrix.astype(np.float32) - vmin) / (vmax - vmin)
    matrix = matrix.astype(np.uint8)
    return matrix


def histogram_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img2 = cdf[img]
    return img2


def contrastLimitedAdaptiveHistogramEqualization(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


def mpcqi(image1, image2, window, L):
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

    return np.mean(pcqi_map)


def sumofCrossCorellation(image0):
    # histogram of original image:
    hist_I = cv.calcHist([image0], [0], None, [256], [0, 256])
    hist_I = np.array(hist_I, dtype=int)
    hist_I = np.reshape(hist_I, (1, 256))

    # hist --value of uniform histogram:
    [s1, s2] = np.shape(image0)
    hist_U_val = s1 * s2 / 256

    # Hist0 [cong thuc 2]
    Lamda = 1000
    hist_O = 1 / (1 + Lamda) * hist_I + Lamda / (1 + Lamda) * hist_U_val
    hist_O = np.array(hist_O, dtype=int)
    hist_O = np.reshape(hist_O, (1, 256))

    # powlog:
    Alpha = 1.1
    list_beta = list(np.arange(1, 2.85, 0.01))

    Corr_values = []
    for beta in list_beta:
        histI_dot1 = np.power(np.log(hist_I + Alpha), beta)  # congthuc 1
        A0 = np.corrcoef(hist_O, hist_I)
        A1 = np.corrcoef(hist_O, histI_dot1)
        Corr_values.append(A0[0, 1] + A1[0, 1])

    Corr_values = np.array(Corr_values)
    return Corr_values


def powTheLogContrastEnhancement(image, alpha, beta):
    lookUpTable = np.empty((1, 256), np.uint8)

    for index in range(256):
        if not (alpha > 1):
            print('invalid alpha')
            break
        lookUpTable[0, index] = np.clip(pow(np.log(index + alpha), beta), 0, 255)  # congthuc 1

    result = cv.LUT(image, lookUpTable)
    return result


ipathB = "/Users/admin/Documents/GitHub/IlluminationNormalization/Input B"
opath_PLCE2 = "/Users/admin/Documents/GitHub/IlluminationNormalization/PLCE2"
opath_PLCE2xHE = "/Users/admin/Documents/GitHub/IlluminationNormalization/PLCE2xHE"
opath_PLCE2xCLAHE = "/Users/admin/Documents/GitHub/IlluminationNormalization/PLCE2xCLAHE"

charsA = 'A/'
charsB = 'B/'

chareA = '_'
chareB = '.jpeg'

path = ipathB
chars = charsB
chare = chareB

# a = cv.imread('Ouput B/HE2/IUNI-U2-camera-thuong-3-201431973247_HE2.png', 0)
# b = cv.imread('Ouput B/ADC/IUNI-U2-camera-thuong-3-201431973247_ADC.png', 0)
# c

for each in glob.iglob(path + '/*.jpeg'):
    imstring = each
    imname = imstring[imstring.find(chars) + 2: imstring.find(chare)]
    he2 = cv.imread('Ouput B/HE2/' + imname + '_HE2.png', 0)
    adc = cv.imread('Ouput B/ADC/' + imname + '_ADC.png', 0)
    print(imname)

    image = cv.imread(each)
    image0 = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv.split(image0)

    dy = sumofCrossCorellation(y_channel)
    ddy = diff(dy) / dx
    index_1 = np.argmax(dy)
    first_beta = tracking_list_beta[index_1]

    dddy = diff(ddy) / dx
    ddddy = diff(dddy) / dx
    index_3 = np.argmax(dddy)
    third_beta = tracking_list_beta[index_3]

    dddddy = diff(ddddy) / dx
    ddddddy = diff(dddddy) / dx
    index_5 = np.argmax(dddddy)
    fifth_beta = tracking_list_beta[index_5]

    last_beta = first_beta
    y_channel = powTheLogContrastEnhancement(y_channel, 3.75, last_beta)

    if (mpcqi(y_channel, he2, gaussiankernel, L) > 5000) | (mpcqi(y_channel, adc, gaussiankernel, L) > 5000):
        last_beta = third_beta
    y_channel = powTheLogContrastEnhancement(y_channel, 3.75, last_beta)

    if (mpcqi(y_channel, he2, gaussiankernel, L) > 200) & (mpcqi(y_channel, adc, gaussiankernel, L) > 1000):
        last_beta = fifth_beta
    y_channel = powTheLogContrastEnhancement(y_channel, 3.75, last_beta)

    # y_channel = rescale(y_channel)
    # y_channel = histogram_equalize(y_channel)
    y_channel_clahe = contrastLimitedAdaptiveHistogramEqualization(y_channel)

    print(last_beta)

    image1 = cv.merge([y_channel, cr_channel, cb_channel])
    image1 = cv.cvtColor(image1, cv.COLOR_YCrCb2RGB)
    cv.imwrite(os.path.join(opath_PLCE2xCLAHE, imname + '_PLCE2.png'), image1)
