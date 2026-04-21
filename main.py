import numpy as np
import joblib
import os
import pandas as pd
import itertools # Kombinasyonları yaratmak için gerekli
from utils import process_folder

# 1. Öznitelik Fonksiyonlarını İçe Aktar
from lbp import extract_lbp
from hog import extract_hog
from glcm import extract_glcm
from dwt import extract_dwt

# 2. Makine Öğrenmesi Modelleri
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

os.makedirs("output/models", exist_ok=True)

# --- DİNAMİK HİBRİT ÖZNİTELİK ÜRETİCİ ---
# Bu fonksiyon, içine verdiğimiz algoritmaları çalıştırıp sonuçları yan yana ekler (Mega-Vektör yapar)
def create_hybrid_extractor(func_list, combo_name):
    def hybrid_func(img):
        # Listedeki her algoritmayı çalıştır ve sonuçları tek vektörde birleştir
        return np.concatenate([f(img) for f in func_list])
    hybrid_func.__name__ = combo_name
    return hybrid_func

# Temel 4 Algoritmamız
base_funcs = {
    "LBP": extract_lbp,
    "HOG": extract_hog,
    "GLCM": extract_glcm,
    "DWT": extract_dwt
}

# --- KOMBİNASYONLARI OLUŞTURMA ---
extractors = {}

# 1. Önce Tekli olanları ekle (LBP, HOG vs.)
for name, func in base_funcs.items():
    extractors[name] = func

# 2. İKİLİ (2-way) Kombinasyonlar (Örn: LBP + DWT, HOG + GLCM)
for combo in itertools.combinations(base_funcs.keys(), 2):
    combo_name = " + ".join(combo)
    func_list = [base_funcs[k] for k in combo]
    extractors[combo_name] = create_hybrid_extractor(func_list, combo_name)

# 3. ÜÇLÜ (3-way) Kombinasyonlar
for combo in itertools.combinations(base_funcs.keys(), 3):
    combo_name = " + ".join(combo)
    func_list = [base_funcs[k] for k in combo]
    extractors[combo_name] = create_hybrid_extractor(func_list, combo_name)

# 4. DÖRTLÜ (4-way) Hepsi Birden
all_4_name = " + ".join(base_funcs.keys())
extractors[all_4_name] = create_hybrid_extractor(list(base_funcs.values()), "Tümü (4'lü)")


# --- SINIFLANDIRICILAR ---
classifiers = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results_matrix =[]

print(f"=== TOPLAM {len(extractors)} FARKLI ÖZNİTELİK KOMBİNASYONU TEST EDİLECEK ===")

# Döngü Başlıyor
for feature_name, extractor_func in extractors.items():
    print(f"\n---[ TEST EDİLİYOR: {feature_name} ] ---")
    
    # 1. Verileri Çıkar
    X_train_real, y_train_real = process_folder("dataset/train/real", 0, extractor_func)
    X_train_fake, y_train_fake = process_folder("dataset/train/fake", 1, extractor_func)
    X_train = np.vstack((X_train_real, X_train_fake))
    y_train = np.hstack((y_train_real, y_train_fake))
    
    X_test_real, y_test_real = process_folder("dataset/test/real", 0, extractor_func)
    X_test_fake, y_test_fake = process_folder("dataset/test/fake", 1, extractor_func)
    X_test = np.vstack((X_test_real, X_test_fake))
    y_test = np.hstack((y_test_real, y_test_fake))
    
    # 2. Ölçeklendirme (Özellikle birleştirilmiş vektörler için HAYATİDİR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    row_results = {"Kombinasyon": feature_name}
    
    # 3. 4 Algoritmayı Eğit ve Test Et
    for clf_name, clf_model in classifiers.items():
        # print(f"  >> {clf_name} eğitiliyor...") # Kalabalık yapmasın diye gizleyebilirsin
        clf_model.fit(X_train_scaled, y_train)
        
        y_pred = clf_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred) * 100
        row_results[clf_name] = f"% {acc:.2f}"
    
    results_matrix.append(row_results)

# --- FİNAL RAPORU ---
print("\n" + "="*80)
print("             GELİŞMİŞ HİBRİT DENEY MATRİSİ SONUÇLARI")
print("="*80)
df_results = pd.DataFrame(results_matrix)
df_results.set_index("Kombinasyon", inplace=True)
pd.set_option('display.max_rows', None) # Tablonun kesilmeden tamamını göstermesi için
print(df_results)
print("="*80)