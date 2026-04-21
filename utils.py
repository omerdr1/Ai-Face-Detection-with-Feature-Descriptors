import cv2
import os
import numpy as np
from tqdm import tqdm

def process_folder(folder_path, label, extractor_func):
    features = []
    labels = []
    file_list =[f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Hangi algoritmanın çalıştığını ekranda göstermek için
    alg_name = extractor_func.__name__
    
    for file_name in tqdm(file_list, desc=f"İşleniyor ({alg_name}): {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Fotoğrafı gri tonlamalı oku ve 128x128 yap
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (128, 128))
            
            # Parametre olarak gelen fonksiyonla (lbp, hog vb.) özellikleri çıkar
            feat = extractor_func(img)
            features.append(feat)
            labels.append(label)
        except Exception as e:
            continue
            
    return np.array(features), np.array(labels)