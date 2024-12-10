import os
import numpy as np
from PIL import Image
import tifffile as tiff
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

def LUPI_gain(ub, lb, x):
    return ((x - lb) / (ub - lb) )*100

def extract_and_save_rgb_ir_eurosat(input_folder):
    """
    Extrae las imágenes RGB (bandas 4, 3, 2) e infrarrojo cercano (banda 8)
    de imágenes TIFF de Sentinel-2 en el directorio de entrada y las guarda
    en carpetas con nombres específicos en el nivel de 'EuroSATallBands'.

    Args:
    - input_folder (str): Carpeta de entrada con imágenes TIFF.
    """
    # Obtener el nombre del subdirectorio (e.g., "River")
    base_dir = os.path.dirname(input_folder)  # Ruta hasta 'EuroSATallBands'
    subfolder_name = os.path.basename(input_folder)  # Nombre del subdirectorio (e.g., "River")
    
    # Crear nuevas carpetas en el nivel de EuroSATallBands
    rgb_folder = os.path.join(base_dir, f"{subfolder_name}RGB")
    ir_folder = os.path.join(base_dir, f"{subfolder_name}NIR")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(ir_folder, exist_ok=True)

    # Procesar todas las imágenes en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # Asegurarse de que solo se procesen archivos .tif
            file_path = os.path.join(input_folder, filename)

            # Leer la imagen TIFF
            tiff_image = tiff.imread(file_path)

            # Verificar que la imagen tiene al menos 13 bandas
            if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 13:
                # Extraer las bandas para RGB (4, 3, 2 en Sentinel-2, índices 3, 2, 1)
                band_4 = tiff_image[:, :, 3]  # Banda Roja (Band 4)
                band_3 = tiff_image[:, :, 2]  # Banda Verde (Band 3)
                band_2 = tiff_image[:, :, 1]  # Banda Azul (Band 2)

                # Normalizar cada banda de forma separada (usando la fórmula band - band.min() / (band.max() - band.min()) * 255)
                band_4_normalized = (band_4 - band_4.min()) / (band_4.max() - band_4.min()) * 255  # Banda Roja
                band_3_normalized = (band_3 - band_3.min()) / (band_3.max() - band_3.min()) * 255  # Banda Verde
                band_2_normalized = (band_2 - band_2.min()) / (band_2.max() - band_2.min()) * 255  # Banda Azul

                # Convertir las bandas normalizadas a tipo uint8 (rango [0, 255])
                band_4_normalized = np.clip(band_4_normalized, 0, 255).astype(np.uint8)
                band_3_normalized = np.clip(band_3_normalized, 0, 255).astype(np.uint8)
                band_2_normalized = np.clip(band_2_normalized, 0, 255).astype(np.uint8)

                # Concatenar las bandas normalizadas en una imagen RGB
                rgb_image = np.stack((band_4_normalized, band_3_normalized, band_2_normalized), axis=-1)  # Orden: Rojo, Verde, Azul

                rgb_output_path = os.path.join(rgb_folder, f"{os.path.splitext(filename)[0]}_RGB.png")
                Image.fromarray(rgb_image).save(rgb_output_path)
                print(f"Guardada RGB: {rgb_output_path}")

                # Extraer la banda infrarroja cercana (banda 8 en Sentinel-2, índice 7)
                ir_band = tiff_image[:, :, 7]  # Índice para la banda 8
                ir_normalized = (ir_band - ir_band.min()) / (ir_band.max() - ir_band.min()) * 255
                ir_normalized = ir_normalized.astype(np.uint8)
                ir_output_path = os.path.join(ir_folder, f"{os.path.splitext(filename)[0]}_NIR.png")
                Image.fromarray(ir_normalized).save(ir_output_path)
                print(f"Guardada NIR: {ir_output_path}")
            else:
                print(f"Saltando archivo (estructura no válida o no tiene suficientes bandas): {file_path}")

    print("Proceso completado.")




def load_image_pairs_with_labels(data_dir, classes, input_size=(64, 64)):
    rgb_paths = []
    nir_paths = []
    labels = []  # Guardar etiquetas: 0 para 'Highway', 1 para 'River'

    for label, class_name in enumerate(classes):
        rgb_folder = os.path.join(data_dir, f'{class_name}RGB')
        nir_folder = os.path.join(data_dir, f'{class_name}NIR')

        for file_name in os.listdir(rgb_folder):
            base_name = "_".join(file_name.split('_')[:-1]) 

            rgb_path = os.path.join(rgb_folder, base_name + '_RGB.png')
            nir_path = os.path.join(nir_folder, base_name + '_NIR.png')

            if os.path.exists(rgb_path) and os.path.exists(nir_path):
                rgb_paths.append(rgb_path)
                nir_paths.append(nir_path)
                labels.append(label)  # Etiqueta basada en el índice de `classes`
            else:
                print('Warning mismatch')
                print(rgb_path)
                print(nir_path)

    # Load and preprocess images
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=input_size)
        return img_to_array(img) / 255.0  # Normalize to [0, 1]

    rgb_images = np.array([preprocess_image(path) for path in rgb_paths])
    nir_images = np.array([preprocess_image(path)[:, :, 0:1] for path in nir_paths])  # Grayscale NIR
    labels = np.array(labels)  # Convertir las etiquetas a numpy array
    
    return rgb_images, nir_images, labels



def concatenate_images(rgb_images, nir_predictions):
    expand = np.repeat(nir_predictions, 3, axis=-1)
    pre_pri = np.concatenate([rgb_images, expand], axis=1)
    return pre_pri  

def expand_array(arr, dim = (1, 64, 64, 1)):
    expanded_array = np.expand_dims(arr, axis=(-1, -2, -3))  
    ex = np.tile(expanded_array, dim) 
    return ex

def loss_TPD(T, beta, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_pred, y_pr )
        bce_r = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        return tf.reduce_mean((1-l)*(bce_r) + l*(tf.math.multiply(d,bce_inst) - beta * tf.math.multiply(1-d, bce_inst))) 
    return loss

def loss_GD(T, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]

        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))
        d1 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        d2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_pr, y_pred)
        return tf.reduce_mean( (1-l)*d1 + l*d2)
    return loss

def loss_MT(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])

    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    sigma = tf.reshape(tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2)), [-1, 1])
    temperature = tf.reshape(tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2)), [-1, 1])

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    l2 = (1/(tf.math.exp(temperature)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre)  + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

    return tf.reduce_mean(l1 + l2)



def loss_MT_TPD(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])
    d = tf.reshape(tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2)), [-1, 1])



    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    sigma = tf.reshape(tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2)), [-1, 1])
    temperature = tf.reshape(tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2)), [-1, 1])

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
    bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
    l2 = (1/(tf.math.exp(temperature)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

    return tf.reduce_mean(l1 + l2)



def loss_MT_PFD(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])

    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    sigma = tf.reshape(tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2)), [-1, 1])
    temperature = tf.reshape(tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2)), [-1, 1])

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce = 0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) 
    l2 = (1/(tf.math.exp(temperature)))*(bce) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

    return tf.reduce_mean(l1 + l2)