import numpy as np
import pywt

def extract_dwt(img):
    # Frekans alanında analiz (checkerboard/dama tahtası artefaktları için)
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    features = []
    # Her frekans bandının ortalamasını ve sapmasını al
    for band in[LL, LH, HL, HH]:
        features.append(np.mean(band))
        features.append(np.std(band))
    return np.array(features)