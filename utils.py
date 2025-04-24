import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score, roc_auc_score, precision_score, recall_score, classification_report

IMG_SIZE = (224, 224)   # Güncellenmiş boyut
BATCH_SIZE = 32 # Her iterasyonda işlenecek olan gruplandırılmış görüntü sayısıdır.
BASE_PATH = "../TUBITAK/datasets/CSV_Multi_Label_Classification"

# Veri Ön İşleme Fonksiyonları
def load_data(subset='train'):
    """CSV dosyasından veri setini yükler ve sütun isimlerini düzeltir"""
    csv_path = os.path.join(BASE_PATH, subset, '_classes.csv')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()# Sütun isimlerindeki boşlukları temizle
    # Her görüntü dosyası için tam dosya yolu oluşturulur
    # lambda fonksiyonu ile her filename değeri için yeni tam yol oluşturulur
    df['filename'] = df['filename'].apply(lambda x: os.path.join(BASE_PATH, subset, x))
    df['no_accident'] = ((df['moderate'] == 0) & (df['severe'] == 0)).astype(int)# no_accident sütununu ekle
    return df

def preprocess_image(image_path, img_size): # img_size parametresi eklendi
    """Görüntüleri yükler, boyutlandırır ve normalize eder"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)  # Güncellenmiş boyut
    return img / 255.0

def create_dataset(df):
    """Fonksiyon şu adımları içerir:
        Görüntü yollarını ve etiketleri TensorFlow dataset'ine dönüştürme.
        Görüntüleri ön işleme.
        Veri setini batch'ler halinde gruplandırma.
        Performans optimizasyonu için prefetch işlemi uygulama.
    """
    images = df['filename'].values
    labels = df[['moderate', 'severe', 'no_accident']].values.astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x, IMG_SIZE), y), num_parallel_calls=tf.data.AUTOTUNE) # IMG_SIZE kullanıldı
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def evaluate_model(model, test_dataset, class_names):
    y_true = []
    y_pred = []
    y_proba = []

    for images, labels in test_dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        trues = np.argmax(labels.numpy(), axis=1)

        y_true.extend(trues)
        y_pred.extend(preds)
        y_proba.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    # Skorlar
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(tf.one_hot(y_true, len(class_names)), y_proba, multi_class='ovr')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Her sınıf için detaylı metrikler
    print("\nSınıf Bazlı Performans:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def predict_single_image(model, image_path, class_names):
    """
    Tek bir görüntü için model tahmini yapar.

    Args:
        model (tf.keras.Model): Eğitimli model.
        image_path (str): Görüntü yolu.
        class_names (list): Sınıf adları.

    Returns:
        predicted_class (str): Tahmin edilen sınıf adı.
        confidence (float): Tahmin güven değeri.
    """
    img = load_img(image_path, target_size= IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_array, axis=0)  # Model girişine uygun hale getirme

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence


def visualize_predictions(model, image_paths, class_names, num_images=5):
    """
    Model tahminlerini rastgele görüntülerle görselleştirir.

    Args:
        model (tf.keras.Model): Eğitimli model.
        image_paths (list or array-like): Görselleştirilecek tüm görüntü yolları.
        class_names (list): Sınıf adları.
        num_images (int): Görselleştirilecek rastgele görüntü sayısı.
    """
    # Görüntü yollarını listeye dönüştür
    image_paths = list(image_paths)

    # Görselleştirmek için rastgele görüntüleri seç
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))

    plt.figure(figsize=(15, 15))
    for i, img_path in enumerate(selected_images):
        # Görüntüyü ve tahmini al
        predicted_class, confidence = predict_single_image(model, img_path, class_names=class_names)
        img = keras.preprocessing.image.load_img(img_path)

        # Alt grafik oluştur ve tahmini ekle
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Tahmin: {predicted_class}\n"
                  f"Güven: {confidence:.2f}")
        plt.axis('off')

    plt.show()
