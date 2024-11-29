# %%
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import utils_image as utim
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/River')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Forest')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/HerbaceousVegetation')

# %%
# Abrir la imagen
image = Image.open("eurosat/EuroSATallBands/ForestRGB/Forest_357_RGB.jpg")


# Obtener las dimensiones
width, height = image.size
print(f"Ancho: {width}, Alto: {height}")
print(image.mode)




# %%

# Define the path to your dataset
dataset_dir = 'eurosat/fr/'

# Collect file paths and labels
data = []
for class_name in ['ForestRGB', 'RiverRGB']:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            label = 0 if class_name == 'ForestRGB' else 1
            data.append((file_path, label))

# Create a DataFrame
df = pd.DataFrame(data, columns=['file_path', 'label'])
print(df.head())  # Verify data



# %%

from sklearn.model_selection import StratifiedKFold

# Inicializar Stratified K-Fold (5 pliegues)
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Extraer rutas de archivos y etiquetas
X = df['file_path']
y = df['label']

# %%

import tensorflow as tf

# Función para cargar y preprocesar las imágenes
def load_and_preprocess_image(file_path, label, img_size=(64, 64)):
    img = tf.io.read_file(file_path)  # Leer la imagen desde el disco
    img = tf.image.decode_jpeg(img, channels=3)  # Decodificar la imagen JPEG
    img = tf.image.resize(img, img_size)  # Redimensionar la imagen
    img = img / 255.0  # Normalizar a rango [0, 1]
    return img, label

# Parámetros de entrenamiento
img_size = (64, 64)  # Tamaño de imagen
batch_size = 1000

# Definir el modelo (por ejemplo, ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Clasificación binaria
model = tf.keras.models.Model(inputs=model.input, outputs=x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  
    patience=5)

# Ciclo de entrenamiento y validación
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}")
    
    # Extract training and test sets for the current fold
    train_files, train_labels = X.iloc[train_idx], y.iloc[train_idx]
    test_files, test_labels = X.iloc[test_idx], y.iloc[test_idx]
    
    # Split 20% from the train set for validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

    # Map the loading and preprocessing function
    train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Train the model for the current fold with EarlyStopping
    model.fit(
        train_dataset, 
        epochs=1, 
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )

    # Optionally evaluate on the test set after training
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
    pre = model.predict(test_dataset)
    err.append(1- accuracy_score(pre))



'''
# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ruta de la imagen
image_path = 'eurosat/EuroSATallBands/RiverRGB/River_357_RGB.jpg'

# Leer la imagen
img = Image.open(image_path)

# Convertir la imagen en un array de NumPy
img_array = np.array(img)

# Mostrar el array de datos
print(img_array)

# Mostrar la imagen
plt.imshow(img_array)
plt.axis('off')  # Desactivar los ejes
plt.show()
'''
# %%




import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Ruta de la imagen TIFF
image_path = 'eurosat/EuroSATallBands/Forest/Forest_1.tif'

tiff_image = tiff.imread(image_path)

# Abrir la imagen TIFF usando rasterio
with rasterio.open(image_path) as src:
    # Ver el número de capas o bandas en la imagen
    num_bands = src.count
    
    # Mostrar cada banda
    for i in range(1, num_bands + 1):
        band = src.read(i)  # Leer la banda
        print(band)
        plt.imshow(band, cmap='gray')  # Mostrar la banda en escala de grises
        plt.title(f"Band {i}")
        plt.colorbar()
        plt.show()

    band_2 = src.read(2)  # Banda Azul (Band 2)
    band_3 = src.read(3)  # Banda Verde (Band 3)
    band_4 = src.read(4)  # Banda Roja (Band 4)

    band_2 = (band_2 / band_2.max()) * 255  # Normalizar a [0, 255]
    band_3 = (band_3 / band_3.max()) * 255
    band_4 = (band_4 / band_4.max()) * 255



    
    # Convertir las bandas a tipo uint8
    band_2 = np.clip(band_2, 0, 255).astype(np.uint8)
    band_3 = np.clip(band_3, 0, 255).astype(np.uint8)
    band_4 = np.clip(band_4, 0, 255).astype(np.uint8)


    band_2 = (band_2 - band_2.min()) / (band_2.max() - band_2.min()) * 255  # Normalizar a [0, 255]
    band_3 = (band_3 - band_3.min()) / (band_3.max() - band_3.min()) * 255
    band_4 = (band_4 - band_4.min()) / (band_4.max() - band_4.min()) * 255

    # Convertir las bandas a tipo uint8
    band_2 = band_2.astype(np.uint8)
    band_3 = band_3.astype(np.uint8)
    band_4 = band_4.astype(np.uint8)

    rgb_image = np.stack((band_4, band_3, band_2), axis=-1)  # Orden: Rojo, Verde, Azul

    # Mostrar la imagen RGB
    plt.imshow(rgb_image)
    plt.title("Imagen RGB de Sentinel-2")
    plt.axis('off')  # Desactivar los ejes
    plt.show()
# %%
