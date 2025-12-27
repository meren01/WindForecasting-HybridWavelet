import matplotlib.pyplot as plt
import seaborn as sns

# --- SENİN VERİLERİN (Tablo 8.1'den alındı) ---
models = ['Hybrid ANN', 'Hybrid RF', 'Baseline RF', 'Baseline ANN', 'Baseline DT', 'Hybrid DT']
rmse_values = [1.2460, 1.3595, 1.3826, 1.3953, 1.7365, 1.8289]

# Renk Ayarı: En iyi model (Hybrid ANN) Yeşil, diğerleri Gri/Kırmızı olsun
colors = ['#2e7d32', '#757575', '#757575', '#c62828', '#bdbdbd', '#bdbdbd']
# (Yeşil: Şampiyon, Kırmızı: Baseline ANN (Rakip), Gri: Diğerleri)

plt.figure(figsize=(10, 6))
bars = plt.bar(models, rmse_values, color=colors, edgecolor='black', alpha=0.8)

# Çubukların üstüne değerleri yazalım
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 4), 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Grafik Süslemeleri
plt.title('Figure 8.2: Model Comparison by RMSE Score (Lower is Better)', fontsize=14, fontweight='bold')
plt.ylabel('RMSE Error Value', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.ylim(0, 2.0)  # Y ekseni sınırı (Görsellik için)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Kaydetme ve Gösterme
plt.tight_layout()
plt.savefig('Figure_8.2_Error_Analysis.png', dpi=300)
plt.show()