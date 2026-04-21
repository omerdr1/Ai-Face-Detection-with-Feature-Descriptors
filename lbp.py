import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(img):
    P = 8
    R = 1
    METHOD = 'uniform'
    lbp = local_binary_pattern(img, P, R, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist