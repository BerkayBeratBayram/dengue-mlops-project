# Dengue Heatmap — Benim Kaggle Çalışmam

Bu repo benim üstünde çalıştığım dengue (haftalık vaka sayısı) tahmin projesini içeriyor. Noteboklarda adım adım izlediğim EDA, feature engineering ve modelleme süreçlerini kod ve açıklama halinde bulabilirsiniz. Aşağıda yaptıklarımı kendi dilimle, pratik ipuçlarıyla özetledim.

## Kısa dosya özeti
- Noteboklar: `01_exploratory_analysis.ipynb`, `02_Feature engineering and modeling.ipynb`
- Veri: `Data/` klasörü (eğitim/test ve örnek submission dosyaları)

## Benim değerlendirme ölçütüm
- Hedef: haftalık vaka sayısı (`total_cases`)
- Ana metrik: Ortalama Mutlak Hata (MAE)

## Yaptıklarım — adım adım

1) Veri yükleme ve ilk kontroller
- CSV'leri `pandas` ile okudum; `info()`, `describe()`, eksik veri tablosu ile veri kalitesine baktım. Özellikle `city`, `year`, `weekofyear` kombinasyonlarını kontrol ettim.

2) Keşifsel Analiz (EDA)
- Şehir bazlı zaman serilerini çizdim, sezonluk desenleri ve olası anomalileri tespit ettim.
- `total_cases` dağılımını inceleyip gerekirse `log1p` dönüşümü denedim.
- Özelliklerin hedefle korelasyonunu ısı haritası ve scatter grafikleri ile kontrol ettim.

3) Feature engineering — öne çıkanlar
- Tarihsel özellikler: `weekofyear`, `month`, `year`, ayrıca mevsimsellik için `sin`/`cos` dönüşümleri kullandım.
- Lag/rolling özellikleri: her şehir için gruplayıp `total_cases` ve meteorolojik değişkenlerin lag'lerini (örn. 1,2,3,4,8 hafta) ve kayan ortalamalarını oluşturdum. (Notebokta örnek kod mevcut.)
- Eksik veriler için şehir bazlı `ffill`/`bfill` uyguladım.
- Kategorik dönüşümler ve etkileşim terimleri (örn. `temp * humidity`) ekledim.

4) Modelleme ve doğrulama
- Zaman serisi yapısını bozmamak için GroupKFold (grup=`city`) ve şehir içi TimeSeriesSplit ile doğrulama yaptım.
- Baseline olarak `lag_1` kullandım; ana modeller olarak `LightGBM` ve `XGBoost` ile çalıştım.

5) Ensemble ve analiz
- CV sırasında out-of-fold tahminleri topladım; basit ağırlıklı ortalama ve stacking ile sonuçları iyileştirdim.
- Önemli özellikleri SHAP ile doğruladım ve veri sızıntısı (leakage) kontrolü yaptım.

6) Test set ve submission
- Test verisi için aynı pipeline'ı uyguladım; lag oluştururken geçmiş haftaları şehir bazında kullandım.
- Tahminleri `clip(0)` ile negatiften arındırıp CSV olarak kaydettim.

## Uygulama notları — kısa ipuçları
- Lag/rolling özelliklerini şehir bazında oluşturun; aksi halde veri sızıntısı olur.
- CV stratejisi zaman bağımlılığını bozmayacak şekilde olmalı; rastgele `KFold` yanıltıcı sonuçlar verir.
- Eksik veri doldururken hedef bilgisini (label) kullanmayın.

## Nasıl çalıştırılır (kısa)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

İsterseniz bu içeriği doğrudan `README.md` ile değiştiririm veya dosyayı sizin için Git ile commit edip push ederim.
