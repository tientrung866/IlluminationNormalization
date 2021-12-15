import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('PLCE2/iPhone11Pro_IMG_0316_575px_PLCE2.png')
# res = ndimage.median_filter(img, )

lower_reso1 = cv.pyrDown(img)
higher_reso1 = cv.pyrUp(lower_reso1)
higher_reso0 = cv.pyrUp(higher_reso1)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(higher_reso0)
plt.show()