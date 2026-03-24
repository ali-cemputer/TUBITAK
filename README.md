# 📹 MOBESE Görüntüleri ile Trafik Kazası ve Şiddet Tespiti (TÜBİTAK 2209-A)

![Python](https://img.shields.io/badge/Python-3.9.23-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.10.0-red?style=for-the-badge&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 🇹🇷 Proje Hakkında

Bu proje, **TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı** kapsamında desteklenmiştir.

Akıllı Ulaşım Sistemleri (AUS) ve şehir güvenliği teknolojileri kapsamında geliştirilen bu çalışmanın temel amacı; şehir izleme kameralarından (MOBESE) elde edilen görüntü akışlarını analiz ederek trafik kazalarını anlık olarak tespit etmek ve kazanın şiddetini **(Kaza Yok, Orta Şiddetli, Ciddi Şiddetli)** sınıflandıran yüksek doğruluklu, derin öğrenme tabanlı bir karar destek sistemi oluşturmaktır.

Geleneksel yöntemlerin aksine, bu proje sadece kazanın varlığını değil, şiddet seviyesini de analiz ederek acil müdahale ekiplerinin (112, İtfaiye) doğru kaynaklarla yönlendirilmesine katkı sağlamayı hedeflemektedir.

## 🎯 Çalışmanın Amacı ve Kapsamı

Trafik kazaları, can ve mal kaybına neden olan en büyük küresel sorunlardan biridir. MOBESE kameralarının yaygınlığına rağmen, bu görüntülerin insan operatörler tarafından sürekli ve dikkatli bir şekilde izlenmesi mümkün değildir. Bu proje şu problemleri çözmeyi hedefler:

* **7/24 Otonom İzleme:** İnsan faktörünü ortadan kaldırarak sürekli kaza denetimi yapmak.
* **Şiddet Analizi:** Kazanın sadece varlığını değil, görüntüdeki hasar boyutuna göre şiddetini (Severity Classification) belirlemek.
* **Yanlış Alarm Minimizesi:** Transfer Learning ve Özgün CNN mimarileri kullanılarak, kaza olmayan durumların kaza gibi algılanmasının önüne geçmek.

* ## 📂 Proje Mimarisi

Proje, problemin farklı boyutlarını ele alan iki temel aşama (Faz) üzerine kurgulanmıştır. Her iki fazda da **Özgün CNN** mimarisi ile literatürdeki SOTA (State-of-the-Art) **Transfer Learning** modelleri karşılaştırılmıştır.

### Dosya Yapısı
```
├───Faz1_MultiLabel_File_Dataset
│   ├───.ipynb_checkpoints
│   ├───best_model_files
│   ├───datasets
│   │   ├───CSV_Multi_Label_Classification
│   │   │   ├───test
│   │   │   ├───train
│   │   │   └───valid
│   │   └───CSV_Multi_Label_Classification_Augmented
│   │       └───train
│   ├───CNN.ipynb
|   ├───EfficientNet.ipynb
|   ├───GoogleNet.ipynb
|   ├───ResNet.ipynb
|   ├───VGG.ipynb
|   ├───utils.py
|
├───Faz2_ACC_SEV_File_Dataset
|   ├───best_model_files
|   └───datasets
|   │    └───CNN_ACC_SEV_V1.v2i.multiclass
|   │        ├───test
|   │        ├───train
|   │        └───valid
|   ├───CNN.ipynb
|   ├───EfficientNet.ipynb
|   ├───GoogleNet.ipynb
|   ├───ResNet.ipynb
|   └───VGG.ipynb
|
└─── README.md
```
## 🚀 Proje Aşamaları ve Metodoloji

Proje, veri seti dengesi ve model optimizasyonu açısından iki temel fazda yürütülmüştür. Her iki fazda da Özgün CNN mimarisi ve Transfer Öğrenme (Transfer Learning) modelleri karşılaştırmalı olarak analiz edilmiştir.

### 🔬 Faz 1: Dengesiz Veri Analizi ve İyileştirme

Projenin ilk aşamasında, Kaggle/Roboflow kaynaklı "CSV_Multi_Label" veri seti kullanılmıştır. Bu aşamanın temel odağı, ham görsel verilerin işlenmesi ve literatürde sıkça karşılaşılan sınıf dengesizliği (class imbalance) probleminin yönetilmesidir.

* **Veri Seti Yapısı:** Toplam 12.122 görüntü. Sınıflar: No Accident, Moderate Accident, Severe Accident.
* **Tespit Edilen Problem:** Veri setinde ciddi bir sınıf dengesizliği tespit edilmiştir. 'Severe' ve 'Moderate' sınıfları baskınken, 'No Accident' sınıfı verinin sadece %2.5'ini oluşturmaktadır.
* **Uygulanan Çözümler:**
    * **Veri Ön İşleme:** Görüntüler 224x224 piksel boyutuna yeniden ölçeklendirilmiş ve piksel yoğunlukları aralığına normalize edilmiştir.
    * **Veri Artırma (Data Augmentation):** Dengesizliği gidermek için sadece azınlık sınıfı olan 'No Accident' örneklerine ImageDataGenerator kullanılarak rastgele döndürme, kaydırma ve yakınlaştırma işlemleri uygulanmış ve sentetik olarak çoğaltılmıştır.
* **Kullanılan Modeller:**
    * **Custom CNN:** Sıfırdan tasarlanan, "Double Convolution" (çiftli evrişim) bloklarına ve artan oranlı Dropout katmanlarına sahip özgün mimari.
    * **Transfer Learning:** VGG19, ResNet50V2, InceptionV3 ve EfficientNetB0 modellerinin konvolüsyonel tabanları dondurularak kullanılmıştır.
* **Faz 1 Sonuçları:** Dengesiz yapıya rağmen EfficientNetB0, uygulanan ön işleme teknikleri sayesinde %88.39 doğrulama doğruluğu ile en iyi performansı göstermiştir.

### 🚀 Faz 2: Kaza Şiddeti Tespiti (Ana Odak)

İkinci aşamada, kaza şiddetinin daha yüksek doğrulukla sınıflandırılması amacıyla dengeli bir yapıya sahip olan "ACC_SEV_V2" veri setine geçiş yapılmıştır. Bu fazda, modellerin gerçek dünya verileri üzerindeki saf performansı ölçülmüştür.

* **Veri Seti Yapısı:** Toplam 7.452 görüntü. Sınıflar arası dağılım (Severe, Moderate, No Accident) birbirine oldukça yakındır (Sınıf başına ~2400-2600 görüntü).
* **Eğitim Stratejisi:**
    * Veri seti dengeli olduğu için bu fazda sentetik veri artırma (Augmentation) işlemine ihtiyaç duyulmamış, modeller ham verinin dengeli yapısı üzerinden eğitilmiştir.
    * **Custom CNN:** Faz 1'deki başarılı mimari korunmuş, öğrenme oranı (learning rate) 0.0005 olarak optimize edilmiştir.
    * **Transfer Learning Standardizasyonu:** Modellerin (VGG19, ResNet50, InceptionV3, EfficientNetB0) öznitelik çıkarma yeteneklerini adil kıyaslamak için hepsine standart bir sınıflandırma bloğu (GlobalAveragePooling -> Dense(256) -> BatchNorm -> Dropout(0.5) -> Softmax) eklenmiştir.
* **Faz 2 Sonuçları:**
    * Dengeli veri seti sayesinde tüm modellerin performansı ciddi oranda artmıştır.
    * EfficientNetB0, %99.44 doğruluk oranı ile en başarılı model olmuştur. Onu %98.87 ile ResNet50V2 takip etmiştir.
    * Bu aşama, veri kalitesi ve dengesinin derin öğrenme modellerindeki kritik rolünü kanıtlamıştır.

