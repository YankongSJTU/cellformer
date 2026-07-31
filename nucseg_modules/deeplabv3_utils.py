import cv2
import numpy as np
import os
import random
import glob
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def histeq2(im, nbr_bins):
    im2 = np.float32(im - im.min()) * np.float32(nbr_bins) / np.float32(im.max() - im.min())
    return im2


def conform(img1, img2):
    """Merge overlapping regions: average where both have values, keep unique regions."""
    _, tmp1 = cv2.threshold(img1, 0, 1, cv2.THRESH_BINARY)
    _, tmp2 = cv2.threshold(img2, 0, 1, cv2.THRESH_BINARY)
    overregion = np.multiply(tmp1, tmp2)
    uniq1 = tmp1 - overregion
    uniq2 = tmp2 - overregion
    tmp4 = (np.multiply(img2, overregion) / 2 + np.multiply(img1, overregion) / 2)
    tmp5 = np.multiply(uniq1, img1)
    tmp6 = np.multiply(uniq2, img2)
    return tmp6 + tmp4 + tmp5


def fillimage(img, h, w):
    newimg = np.zeros([h, w, 3])
    h1 = img.shape[0]
    w1 = img.shape[1]
    newimg[0:h1, 0:w1] = img
    return newimg


def cropimg(img, h, w):
    newimg = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return newimg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
