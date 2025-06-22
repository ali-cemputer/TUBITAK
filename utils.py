import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, classification_report

# Global Sabitler
IMG_SIZE = (224, 224)  # Görüntülerin yeniden boyutlandırılacağı piksel boyutu.
BATCH_SIZE = 32  # Model eğitimi sırasında kullanılacak mini-batch boyutu.
ORIGINAL_DATA_PATH = "../TUBITAK/datasets/CSV_Multi_Label_Classification"  # Orijinal veri setinin ana yolu.
AUGMENTED_DATA_PATH = "../TUBITAK/datasets/CSV_Multi_Label_Classification_Augmented"  # Artırılmış veri setinin kaydedileceği ana yol.


# --- Veri Yükleme ve Ön İşleme Fonksiyonları ---

def load_data(subset='train', base_path=ORIGINAL_DATA_PATH):
    """
    Belirtilen alt küme için CSV dosyasından veri yükler, sütun isimlerini temizler
    ve dosya yollarını mutlak yollara dönüştürür.
    'no_accident' sınıfı için ikili etiket sütunu ekler.

    Args:
        subset (str): Yüklenecek veri alt kümesi ('train', 'valid' veya 'test').
        base_path (str): Veri setinin ana dizin yolu.

    Returns:
        pd.DataFrame: Yüklenen ve işlenen veri içeren DataFrame.
    """
    csv_path = os.path.join(base_path, subset, '_classes.csv')

    # CSV dosyasını oku (varsayılan kodlamayla dene)
    df = pd.read_csv(csv_path, encoding='utf-8')

    df.columns = df.columns.str.strip()  # Sütun isimlerindeki boşlukları temizle

    # 'filename' sütunundaki yolların zaten tam yol olup olmadığını kontrol et
    # ve gerekirse tam yolu oluştur
    if not df.empty and df['filename'].iloc[0].startswith(base_path):
        df['filename'] = df['filename'].apply(os.path.normpath)  # Yolları işletim sistemine uygun normalleştir
    else:
        df['filename'] = df['filename'].apply(lambda x: os.path.normpath(os.path.join(base_path, subset, x)))

    # 'no_accident' etiketini oluştur
    df['no_accident'] = ((df['moderate'] == 0) & (df['severe'] == 0)).astype(int)
    return df


def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Görüntü dosyasını yükler, belirtilen boyuta yeniden boyutlandırır ve pikselleri normalize eder (0-1 aralığına).
    TensorFlow grafik uyumlu hale getirildi.

    Args:
        image_path (tf.Tensor): Görüntü dosyasının Tensor formatında yolu.
        img_size (tuple): Görüntünün yeniden boyutlandırılacağı (genişlik, yükseklik) tuple.

    Returns:
        tf.Tensor: Ön işlenmiş görüntü Tensor'u.
    """
    # Doğrudan TensorFlow fonksiyonlarını kullan
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img


def create_dataset(df, batch_size=BATCH_SIZE):
    """
    Pandas DataFrame'den TensorFlow Dataset nesnesi oluşturur.
    Görüntüleri ön işler, veriyi batch'lere ayırır ve prefetch ile performansı artırır.

    Args:
        df (pd.DataFrame): Görüntü yolları ve etiketleri içeren DataFrame.
        batch_size (int): Dataset'in batch boyutu.

    Returns:
        tf.data.Dataset: Hazırlanmış TensorFlow Dataset nesnesi.
    """
    images = df['filename'].values
    labels = df[['moderate', 'severe', 'no_accident']].values.astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # Görüntüleri paralel olarak ön işleme
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Veri hazır olana kadar beklemeden bir sonraki adımı hazırla
    return dataset


# --- Model Değerlendirme ve Görselleştirme Fonksiyonları ---

def evaluate_model(model, test_dataset, class_names):
    """
    Eğitimli bir modelin performansını test veri seti üzerinde değerlendirir.
    Doğruluk, F1-Skor, ROC-AUC gibi metrikleri hesaplar ve karmaşıklık matrisi ile
    sınıf bazlı performans raporunu gösterir.

    Args:
        model (tf.keras.Model): Değerlendirilecek eğitilmiş model.
        test_dataset (tf.data.Dataset): Modelin test edileceği veri seti.
        class_names (list): Sınıf etiketlerinin adlarını içeren liste.
    """
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


def predict_single_image(model, image_path, class_names, img_size=IMG_SIZE):
    """
    Tek bir görüntü için model tahmini yapar ve tahmin edilen sınıf ile güven değerini döndürür.

    Args:
        model (tf.keras.Model): Eğitimli model.
        image_path (str): Tahmin edilecek görüntünün dosya yolu.
        class_names (list): Sınıf adlarının listesi.
        img_size (tuple): Görüntünün modelin beklediği boyutu (genişlik, yükseklik).

    Returns:
        tuple: (predicted_class (str), confidence (float)) Tahmin edilen sınıf adı ve güven değeri.
    """
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence


def visualize_predictions(model, image_paths, class_names, num_images=5, img_size=IMG_SIZE):
    """
    Model tahminlerini rastgele seçilen görüntülerle görselleştirir.
    Görüntüleri ve modelin tahminini, güven değeriyle birlikte gösterir.

    Args:
        model (tf.keras.Model): Eğitimli model.
        image_paths (list or np.array): Görselleştirilecek tüm görüntü yollarının listesi.
        class_names (list): Sınıf adlarının listesi.
        num_images (int): Görselleştirilecek rastgele görüntü sayısı.
        img_size (tuple): Görüntünün modelin beklediği boyutu (genişlik, yükseklik).
    """
    image_paths_list = list(image_paths)
    selected_images = random.sample(image_paths_list, min(num_images, len(image_paths_list)))

    plt.figure(figsize=(num_images * 4, 4))
    for i, img_path in enumerate(selected_images):
        predicted_class, confidence = predict_single_image(model, img_path, class_names, img_size)
        img = load_img(img_path)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Tahmin: {predicted_class}\nGüven: {confidence:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Veri Artırma Fonksiyonları ---

def augment_image_data_gen(image_array, data_gen):
    """
    Tek bir görüntüyü Keras ImageDataGenerator kullanarak artırır.

    Args:
        image_array (np.array): Artırılacak görüntünün NumPy array'i (RGB formatında).
        data_gen (tf.keras.preprocessing.image.ImageDataGenerator): KullanılacakImageDataGenerator nesnesi.

    Returns:
        np.array: Artırılmış görüntünün NumPy array'i.
    """
    # ImageDataGenerator, (batch_size, height, width, channels) şeklinde giriş bekler.
    image_expanded = np.expand_dims(image_array, 0)
    aug_iter = data_gen.flow(image_expanded, batch_size=1)
    aug_image = next(aug_iter)[0].astype('uint8')
    return aug_image


def augment_and_save_data(df, subset_name,
                          no_accident_aug_size=0,
                          moderate_aug_size=0,
                          severe_aug_size=0,
                          output_base_path=AUGMENTED_DATA_PATH):
    """
    Veri çerçevesindeki görüntüleri belirli oranlarda artırır ve yeni bir dizine kaydeder.
    Bu işlem, orijinal görüntüleri ve artırılmış versiyonlarını içerir.
    Veri artırma işlemi sırasında sınıf dengesi sağlanabilir.

    Args:
        df (pd.DataFrame): Orijinal veri içeren DataFrame.
        subset_name (str): İşlem yapılan veri alt kümesinin adı ('train', 'valid', 'test').
        no_accident_aug_size (int): Her 'No Accident' görüntüsü için kaç adet artırılmış kopya oluşturulacağı.
        moderate_aug_size (int): Her 'Moderate Accident' görüntüsü için kaç adet artırılmış kopya oluşturulacağı.
        severe_aug_size (int): Her 'Severe Accident' görüntüsü için kaç adet artırılmış kopya oluşturulacağı.
        output_base_path (str): Artırılmış verilerin kaydedileceği ana dizin yolu.

    Returns:
        pd.DataFrame: Artırılmış ve orijinal verilerin birleşimini içeren yeni DataFrame.
    """
    augmented_rows = []

    output_subset_path = os.path.join(output_base_path, subset_name)
    os.makedirs(output_subset_path, exist_ok=True)  # Dizini zaten varsa oluşturma/varlığını kontrol etme

    # Veri artırma için ImageDataGenerator tanımla
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
        brightness_range=[0.8, 1.2], fill_mode='nearest'
    )

        # Her bir orijinal görüntüyü işle
    for index, row in df.iterrows():
        original_img_path = row['filename']
        base_name = os.path.basename(original_img_path)
        new_original_path = os.path.join(output_subset_path, base_name)

        # Orijinal görüntüyü yeni konuma kopyala (sadece dosya yoksa veya boyutu 0 ise)
        if not os.path.exists(new_original_path) or os.path.getsize(new_original_path) == 0:
            original_img_bgr = cv2.imread(original_img_path)
            cv2.imwrite(new_original_path, original_img_bgr)

        # Kopyalanan orijinal görüntüyü DataFrame'e ekle
        original_row_copy = row.copy()
        original_row_copy['filename'] = new_original_path
        augmented_rows.append(original_row_copy)

        # Görüntüyü RGB'ye dönüştür (artırma için)
        original_img_rgb = cv2.cvtColor(cv2.imread(new_original_path), cv2.COLOR_BGR2RGB)  # Kopyalanan resmi oku

        # Hangi sınıfın kaç kez artırılacağını belirle
        aug_count = 0
        if (row['no_accident'] == 1) and no_accident_aug_size > 0:
            aug_count = no_accident_aug_size
        elif (row['moderate'] == 1) and (row['severe'] == 0) and moderate_aug_size > 0:
            aug_count = moderate_aug_size
        elif (row['severe'] == 1) and (row['moderate'] == 0) and severe_aug_size > 0:
            aug_count = severe_aug_size

        # Görüntüleri artır ve kaydet
        for i in range(aug_count):
            augmented_image_array = augment_image_data_gen(original_img_rgb, datagen)
            aug_filename = f"{os.path.splitext(base_name)[0]}_aug_{i}.jpg"
            new_aug_path = os.path.join(output_subset_path, aug_filename)

            cv2.imwrite(new_aug_path, cv2.cvtColor(augmented_image_array, cv2.COLOR_RGB2BGR))

            new_row = row.copy()
            new_row['filename'] = new_aug_path
            augmented_rows.append(new_row)

    final_df = pd.DataFrame(augmented_rows)

    # _classes.csv dosyasını kaydet
    final_csv_path = os.path.join(output_subset_path, '_classes.csv')
    final_df[['filename', 'moderate', 'severe', 'no_accident']].to_csv(final_csv_path, index=False, encoding='utf-8')

    # Son sınıf dağılımını yazdır
    print(f"--- Artırılmış '{subset_name}' sınıf dağılımı (Yeni Oluşturulan DF'den): ---")
    print(
        f"Moderate Accident: {final_df['moderate'].sum()}, Severe Accident: {final_df['severe'].sum()}, No Accident: {final_df['no_accident'].sum()}")

    return final_df