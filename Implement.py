import os
import time
import numpy as np
import cv2 as cv
import glob

ipath = "/Users/admin/Documents/GitHub/IlluminationNormalization/Input A"
opath = "/Users/admin/Documents/GitHub/IlluminationNormalization/HE1"

# region FUNCTION
def rescale(matrix):
    vmin = matrix.min()
    vmax = matrix.max()
    matrix = 255 * (matrix.astype(np.float32) - vmin) / (vmax - vmin)
    matrix = matrix.astype(np.uint8)
    return matrix


def histogram(matrix):
    m, n = matrix.shape
    pdf = np.zeros(256)
    for index in range(0, 255):
        pdf[index] = sum(sum(matrix == index)) / (m * n)
    return pdf


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


def naive_gamma_correction(matrix, gamma):
    return np.power(matrix / 255.0, gamma)


def gammaCorrection(img_original, gamma):
    # [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img_original, lookUpTable)
    # [changing-contrast-brightness-gamma-correction]
    return res


def adaptiveGammaCorrection(image, gamma):
    pre = gammaCorrection(image, gamma)
    PDF = histogram(pre)

    lookUpTable = np.empty((1, 256), np.uint8)

    for index in range(256):
        CDF = sum(PDF[:index])
        gamma = 1 - CDF
        lookUpTable[0, index] = np.clip(pow(index / 255.0, gamma) * 255.0, 0, 255)

    post = cv.LUT(pre, lookUpTable)
    return post


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
    list_beta = list(np.arange(1, 50, 0.1))

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


# endregion
chars = '/'
chare = '_'

# myipath = os.path.join(ipath, '/*.png')
for each in glob.iglob(ipath + '/*.png'):
    imstring = each
    imname = imstring[imstring.find(chars)+1 : imstring.find(chare)]

    image = cv.imread(each)
    image0 = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv.split(image0)
    y_channel = histogram_equalize(y_channel)

    image1 = cv.merge([y_channel, cr_channel, cb_channel])
    image1 = cv.cvtColor(image1, cv.COLOR_YCrCb2RGB)
    cv.imwrite(os.path.join(opath, '/', imname, '_HE1.png'), image1)

