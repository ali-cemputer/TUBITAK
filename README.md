# 📹 MOBESE Görüntüleri ile Trafik Kazası ve Şiddet Tespiti (TÜBİTAK 2209-A)

![Python](https://img.shields.io/badge/Python-3.9.23-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.10.0-red?style=for-the-badge&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 🇹🇷 Proje Hakkında

Bu proje, **TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı** kapsamında desteklenmiştir. 

**📌 Yayın Bilgisi**
4. Uluslararası İYONYA Konferansı'nda uluslararası bildiri olarak yayımlanmıştır. Projenin amacı, detayları ve sonuçları hakkında kapsamlı bilgiye [ISARC Konferans Kitabı](https://www.isarconference.org/_files/ugd/6dc816_9a032037ee0b4e4b908bc719cafb587c.pdf) üzerinden "Ali Cem Duran" araması yaparak ulaşabilirsiniz.

Akıllı Ulaşım Sistemleri (AUS) ve şehir güvenliği teknolojileri kapsamında geliştirilen bu çalışmanın temel amacı; şehir izleme kameralarından (MOBESE) elde edilen görüntü akışlarını analiz ederek trafik kazalarını anlık olarak tespit etmek ve kazanın şiddetini **(Kaza Yok, Orta Şiddetli, Ciddi Şiddetli)** sınıflandıran yüksek doğruluklu, derin öğrenme tabanlı bir karar destek sistemi oluşturmaktır.

Geleneksel yöntemlerin aksine, bu proje sadece kazanın varlığını değil, şiddet seviyesini de analiz ederek acil müdahale ekiplerinin (112, İtfaiye) doğru kaynaklarla yönlendirilmesine katkı sağlamayı hedeflemektedir.

* ## 📂 Proje Mimarisi

Proje, problemin farklı boyutlarını ele alan iki temel aşama (Faz) üzerine kurgulanmıştır. Her iki fazda da **Özgün CNN** mimarisi ile **Transfer Learning** modelleri karşılaştırılmıştır.

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


