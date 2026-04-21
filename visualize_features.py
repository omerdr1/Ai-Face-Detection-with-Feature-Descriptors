import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
import os
from scipy.stats import entropy

os.makedirs("output/visualizations", exist_ok=True)

def visualize_algorithms(image_path):
    print(f"\n[{os.path.basename(image_path)}] için İstatistiksel Görselleştirme başlatılıyor...")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Hata: Fotoğraf bulunamadı!")
        return
    img = cv2.resize(img, (256, 256))

    # ==========================================
    # 1. LBP (DOKU VE ENTROPİ ANALİZİ)
    # ==========================================
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    
    # İSTATİSTİK HESAPLAMA: Doku Karmaşıklığı (Shannon Entropy)
    # Entropi düşükse cilt "plastik" gibi pürüzsüzdür (AI genelde böyledir)
    tex_entropy = entropy(hist, base=2)
    dominant_pattern_ratio = np.max(hist) * 100 # En çok tekrar eden dokunun oranı

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(lbp, cmap='gray')
    plt.title("LBP Doku Haritası")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    bars = plt.hist(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), color='#3498db', alpha=0.8, density=True)
    plt.title(f"LBP Dağılımı\nKarmaşıklık (Entropi): {tex_entropy:.2f} | Baskın Doku: %{dominant_pattern_ratio:.1f}")
    
    plt.tight_layout()
    plt.savefig("output/visualizations/1_LBP_Analizi_Oranli.png", dpi=300)
    plt.close()
    print(" > 1. LBP görseli (Entropi oranlarıyla) kaydedildi.")

    # ==========================================
    # 2. HOG (KENAR VE GRADYAN ENERJİSİ)
    # ==========================================
    features, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True)
    
    # İSTATİSTİK HESAPLAMA: Gradyan Yoğunluğu
    total_edge_energy = np.sum(features)
    active_edge_ratio = (np.count_nonzero(features > 0.1) / len(features)) * 100

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray', vmax=hog_image.max(), vmin=0)
    plt.title(f"HOG Vektörleri\nToplam Kenar Enerjisi: {total_edge_energy:.1f}\nBelirgin Hat Oranı: %{active_edge_ratio:.1f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("output/visualizations/2_HOG_Analizi_Oranli.png", dpi=300)
    plt.close()
    print(" > 2. HOG görseli (Enerji oranlarıyla) kaydedildi.")

    # ==========================================
    # 3. DWT (FREKANS VE GÜRÜLTÜ ORANLARI - EN ÖNEMLİSİ)
    # ==========================================
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    # İSTATİSTİK HESAPLAMA: Enerji Oranları
    e_LL = np.sum(LL**2)
    e_LH = np.sum(LH**2)
    e_HL = np.sum(HL**2)
    e_HH = np.sum(HH**2)
    total_energy = e_LL + e_LH + e_HL + e_HH
    
    # Düşük Frekans (Ana Görüntü) vs Yüksek Frekans (Gürültü/Detay)
    noise_ratio = ((e_LH + e_HL + e_HH) / total_energy) * 100
    hh_ratio = (e_HH / total_energy) * 100 # Sırf dama tahtası (checkerboard) gürültüsü
    
    plt.figure(figsize=(12, 10))
    fig_title = f"DWT Frekans Analizi\nToplam Yüksek Frekans (Gürültü/Detay) Oranı: %{noise_ratio:.4f}"
    plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)

    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.title(f"LL (Temel)\nEnerji Yükü: %{(e_LL/total_energy)*100:.2f}")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(LH, cmap='gray')
    plt.title(f"LH (Yatay Kenarlar)\nStandart Sapma: {np.std(LH):.2f}")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(HL, cmap='gray')
    plt.title(f"HL (Dikey Kenarlar)\nStandart Sapma: {np.std(HL):.2f}")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(HH, cmap='gray')
    plt.title(f"HH (Çapraz Gürültü/Artefakt)\nToplam İçindeki Oranı: %{hh_ratio:.5f}\nStd Sapma: {np.std(HH):.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("output/visualizations/3_DWT_Analizi_Oranli.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" > 3. DWT görseli (Gürültü/Artefakt oranlarıyla) kaydedildi.")

    # ==========================================
    # 4. GLCM (İSTATİSTİKSEL MATRİS VE SKORLAR)
    # ==========================================
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    glcm_plot = np.log1p(glcm[:, :, 0, 0]) 
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(glcm_plot, cmap='hot')
    plt.title(f"GLCM (Piksel Komşuluk Olasılıkları)\n"
              f"Kontrast (Pürüzlülük): {contrast:.1f} | Homojenlik (Pürüzsüzlük): {homogeneity:.3f}\n"
              f"Enerji (Tekrar Oranı): {energy:.3f} | Korelasyon: {correlation:.3f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("output/visualizations/4_GLCM_Analizi_Oranli.png", dpi=300)
    plt.close()
    print(" > 4. GLCM görseli (Pürüzsüzlük/Tekrar oranlarıyla) kaydedildi.")
    print("\nTüm analiz görselleri 'output/visualizations' klasörüne kaydedilmiştir!")

TEST_FOTOGRAFI = "dataset/train/fake/easy_4_0011.jpg" 
visualize_algorithms(TEST_FOTOGRAFI)