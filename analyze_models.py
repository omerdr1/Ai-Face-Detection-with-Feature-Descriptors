import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Kendi yazdığımız fonksiyonları içe aktarıyoruz
from utils import process_folder
from dwt import extract_dwt  # DWT özniteliğini kullanacağız (İstersen glcm, lbp yapabilirsin)

# Rapor grafikleri için klasör
os.makedirs("output/analysis_plots", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=== GERÇEK VERİ SETİ İLE ANALİZ BAŞLIYOR ===")

# =====================================================================
# 1. GERÇEK VERİ SETİNİ YÜKLEME
# =====================================================================
print("1. Eğitim verileri işleniyor...")
X_train_real, y_train_real = process_folder("dataset/train/real", 0, extract_dwt)
X_train_fake, y_train_fake = process_folder("dataset/train/fake", 1, extract_dwt)
X_train = np.vstack((X_train_real, X_train_fake))
y_train = np.hstack((y_train_real, y_train_fake))

print("\n2. Test verileri işleniyor...")
X_test_real, y_test_real = process_folder("dataset/test/real", 0, extract_dwt)
X_test_fake, y_test_fake = process_folder("dataset/test/fake", 1, extract_dwt)
X_test = np.vstack((X_test_real, X_test_fake))
y_test = np.hstack((y_test_real, y_test_fake))

# Veriyi Ölçeklendir (Modellerin adil çalışması için şart)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 2. MODELLERİN TANIMLANMASI VE EĞİTİMİ
# =====================================================================
models = {
    "SVM (Doğrusal)": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

trained_models = {}
print("\n3. Modeller eğitiliyor ve test ediliyor...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"   > {name} tamamlandı.")

# =====================================================================
# ANALİZ 1: KARMAŞIKLIK MATRİSLERİ (CONFUSION MATRIX)
# =====================================================================
print("\n4. Grafikler oluşturuluyor...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Algoritmaların Karar Analizi (Karmaşıklık Matrisi - Gerçek Veri)', fontsize=16)
axes = axes.flatten()

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Gerçek (0)', 'Sahte (1)'], 
                yticklabels=['Gerçek (0)', 'Sahte (1)'],
                annot_kws={"size": 14})
    axes[i].set_title(name, fontsize=14)
    axes[i].set_ylabel('Asıl Olan (True Label)')
    axes[i].set_xlabel('Modelin Tahmini (Predicted)')

plt.tight_layout()
plt.savefig("output/analysis_plots/1_Confusion_Matrices_RealData.png", dpi=300)
plt.close()

# =====================================================================
# ANALİZ 2: ÖZNİTELİK ÖNEMİ (FEATURE IMPORTANCE)
# =====================================================================
# Eğer DWT kullandıysak, döndürdüğü 8 elemanın adları şunlardır:
feature_names =['LL (Temel)_Ortalama', 'LL_StandartSapma', 
                 'LH (Yatay)_Ortalama', 'LH_StandartSapma', 
                 'HL (Dikey)_Ortalama', 'HL_StandartSapma', 
                 'HH (Çapraz)_Ortalama', 'HH_StandartSapma']

rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_

# Güvenlik önlemi: Eğer çıkarılan özellik sayısı isimlerle uyuşmuyorsa hata vermesin
if len(importances) == len(feature_names):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    sorted_features =[feature_names[i] for i in indices]

    sns.barplot(x=importances[indices], y=sorted_features, palette="viridis")
    plt.title('Random Forest: Kendi Veri Setimizde En Çok Neye Baktı?', fontsize=14)
    plt.xlabel('Önem Derecesi (Katsayı)')
    plt.ylabel('DWT Frekans Bandı')

    plt.tight_layout()
    plt.savefig("output/analysis_plots/2_Feature_Importance_RealData.png", dpi=300)
    plt.close()
else:
    print(f"Uyarı: Öznitelik isimleri ({len(feature_names)}) ile veri boyutu ({len(importances)}) uyuşmuyor. Çubuk grafiği atlandı.")

# =====================================================================
# ANALİZ 3: ROC EĞRİLERİ (ROC CURVES)
# =====================================================================
plt.figure(figsize=(10, 8))

for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1],[0, 1], color='gray', lw=2, linestyle='--', label='Rastgele Tahmin (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (Gerçek insana sahte deme)')
plt.ylabel('Doğru Pozitif Oranı (Sahteyi doğru bulma)')
plt.title('Modellerin Performans Karşılaştırması (Kendi Veri Setimiz)', fontsize=14)
plt.legend(loc="lower right", fontsize=12)

plt.tight_layout()
plt.savefig("output/analysis_plots/3_ROC_Curves_RealData.png", dpi=300)
plt.close()

print("\nTüm işlemler bitti! Grafikler 'output/analysis_plots' klasörüne kaydedildi.")