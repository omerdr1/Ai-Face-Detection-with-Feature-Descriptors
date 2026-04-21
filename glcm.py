import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm(img):
    # İstatistiksel doku analizi (Kontrast, Korelasyon, Enerji, Homojenlik)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])