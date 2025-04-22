import os
import pandas as pd
import tensorflow as tf

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