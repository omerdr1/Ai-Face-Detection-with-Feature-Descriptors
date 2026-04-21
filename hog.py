from skimage.feature import hog

def extract_hog(img):
    # Geometrik köşe ve kenar analizleri için
    features = hog(img, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)
    return features