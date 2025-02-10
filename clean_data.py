
# %%
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import utils_preprocess as utim
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import shutil



#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/River')
#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Forest')
#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/HerbaceousVegetation')
#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/AnnualCrop')
#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/PermanentCrop')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Highway')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Pasture')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Industrial')
utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Residential')


utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/River')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/Forest')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/HerbaceousVegetation')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/AnnualCrop')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/PermanentCrop')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/Highway')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/Pasture')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/Industrial')
utim.extract_and_save_rgb_nir_swir_eurosat('eurosat/EuroSATallBands/Residential')


# %%
#----------------------------------------------------------------
#MODELO LOWER
#----------------------------------------------------------------

#var = ['AnnualCrop', 'PermanentCrop']
var  = ['Industrial', 'Residential']
var  = ['Highway', 'River']
var  = ['Pasture', 'Forest']
var = ['AnnualCrop', 'PermanentCrop']


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#DIVISION TRAIN-TEST LOWER
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX


import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Definir las rutas de las carpetas de las clases
root_folder = 'eurosat/EuroSATallBands/'  # Carpeta raíz donde están RGB y NIR
classes = [var[0] + 'RGB', var[1] + 'RGB']  # Carpetas RGB
nir_classes = [var[0] + 'NIR', var[1] + 'NIR']  # Carpetas NIR asociadas

# Preparar listas para almacenar las rutas de las imágenes RGB, NIR y sus etiquetas
rgb_paths = []
nir_paths = []
labels = []

# Recorrer las carpetas de clases y recopilar las rutas de las imágenes RGB y NIR
for label, class_name in enumerate(classes):
    rgb_class_folder = os.path.join(root_folder, class_name)
    nir_class_folder = os.path.join(root_folder, nir_classes[label])

    for image_file in os.listdir(rgb_class_folder):
        if image_file.endswith('.png'):  # Asegurarnos de solo procesar imágenes
            rgb_path = os.path.join(rgb_class_folder, image_file)
            nir_path = os.path.join(nir_class_folder, image_file.replace('RGB', 'NIR'))  # Asociar imagen NIR
            if os.path.exists(nir_path):  # Verificar que la imagen NIR existe
                rgb_paths.append(rgb_path)
                nir_paths.append(nir_path)
                labels.append(label)  # Etiqueta de la clase (0 o 1)

# Convertir las listas a numpy arrays
rgb_paths = np.array(rgb_paths)
nir_paths = np.array(nir_paths)
labels = np.array(labels)

# Dividir en train (80%) y test (20%) usando un 80/20
X_train_rgb, X_test_rgb, X_train_nir, X_test_nir, y_train, y_test = train_test_split(
    rgb_paths, nir_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Crear directorios para guardar las particiones de train y test
train_dir = 'eurosat/split/train'
test_dir = 'eurosat/split/test'

for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

for class_name in nir_classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Mover las imágenes RGB y NIR a sus respectivos directorios de train y test
for i, rgb_path in enumerate(X_train_rgb):
    label = y_train[i]
    rgb_class_name = classes[label]
    nir_class_name = nir_classes[label]

    # Copiar RGB
    shutil.copy(rgb_path, os.path.join(train_dir, rgb_class_name, os.path.basename(rgb_path)))
    # Copiar NIR
    shutil.copy(X_train_nir[i], os.path.join(train_dir, nir_class_name, os.path.basename(X_train_nir[i])))

for i, rgb_path in enumerate(X_test_rgb):
    label = y_test[i]
    rgb_class_name = classes[label]
    nir_class_name = nir_classes[label]

    # Copiar RGB
    shutil.copy(rgb_path, os.path.join(test_dir, rgb_class_name, os.path.basename(rgb_path)))
    # Copiar NIR
    shutil.copy(X_test_nir[i], os.path.join(test_dir, nir_class_name, os.path.basename(X_test_nir[i])))

print(f"Se han creado los conjuntos de train y test con {len(X_train_rgb)} imágenes RGB para entrenamiento y {len(X_test_rgb)} para prueba.")




# %%
'''
#----------------------------------------------------------------
#MODELO UPPER
#----------------------------------------------------------------

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#JUNTAR RGB E IR
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Directorio raíz donde se encuentran las carpetas
root_folder = 'eurosat/EuroSATallBands/'  

# Función para combinar RGB y NIR en una sola imagen (4 canales)
def combine_rgb_nir(rgb_image_path, nir_image_path, output_image_path):
    # Cargar las imágenes RGB y NIR (como imágenes PNG)
    rgb_image = Image.open(rgb_image_path).convert('RGB')  # Asegurarse de que la imagen RGB tenga 3 canales
    nir_image = Image.open(nir_image_path).convert('L')  # Asegurarse de que la imagen NIR tenga 1 canal (escala de grises)

    # Asegúrate de que las imágenes RGB y NIR tengan el mismo tamaño
    rgb_image = rgb_image.resize((nir_image.width, nir_image.height))

    # Convertir las imágenes a arrays de numpy
    rgb_array = np.array(rgb_image)
    nir_array = np.array(nir_image)

    # Replicar la imagen NIR para que tenga 3 canales (para que sea compatible con la estructura RGB)
    nir_array_3channels = np.stack([nir_array] * 3, axis=-1)  # Convertir NIR a 3 canales

    # Combinar las imágenes: RGB arriba, NIR abajo
    combined_image = np.concatenate((rgb_array, nir_array_3channels), axis=0)

    # Convertir el array combinado de nuevo a una imagen
    combined_image_pil = Image.fromarray(combined_image.astype(np.uint8))

    # Guardar la imagen combinada en formato PNG
    combined_image_pil.save(output_image_path, format='PNG')

    print(f"Imagen guardada en: {output_image_path}")

# Lista de clases (carpetas) que contienen las imágenes RGB y NIR


# Recorrer las clases (subcarpetas)
for class_name in var:
    # Crear nuevas carpetas para las imágenes combinadas (PI)
    class_output_folder = os.path.join(root_folder, f"{class_name}PI")
    os.makedirs(class_output_folder, exist_ok=True)

    # Acceder a las carpetas RGB y NIR
    rgb_folder = os.path.join(root_folder, f"{class_name}RGB")
    nir_folder = os.path.join(root_folder, f"{class_name}NIR")

    # Recorrer las imágenes en las carpetas RGB
    for rgb_filename in os.listdir(rgb_folder):
        if rgb_filename.endswith("_RGB.png"):  # Filtrar solo las imágenes RGB
            # Extraer el número de la imagen (por ejemplo, 1 de Forest_1_RGB.png)
            image_number = rgb_filename.split('_')[1]

            # Construir el nombre del archivo NIR correspondiente (ej. Forest_1_NIR.png)
            nir_filename = f"{class_name}_{image_number}_NIR.png"

            # Comprobar si el archivo NIR existe
            nir_image_path = os.path.join(nir_folder, nir_filename)
            if os.path.exists(nir_image_path):
                rgb_image_path = os.path.join(rgb_folder, rgb_filename)

                # Crear el nombre del archivo para la imagen combinada
                output_image_path = os.path.join(class_output_folder, f"{class_name}_{image_number}_combined.png")

                # Combinar las imágenes y guardarlas
                combine_rgb_nir(rgb_image_path, nir_image_path, output_image_path)

print("Proceso completado.")


# %%

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#DIVISION TRAIN-TEST UPPER
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# Definir las rutas de las carpetas de las clases
root_folder = 'eurosat/EuroSATallBands/'   # Carpeta raíz donde están class_1 y class_2
classes = [var[0] + 'PI', var[1] +'PI']  # Listar las subcarpetas (class_1, class_2)

# Preparar listas para almacenar las rutas de las imágenes y sus etiquetas
image_paths = []
labels = []

# Recorrer las carpetas de clases y recopilar las rutas de las imágenes
for label, class_name in enumerate(classes):
    class_folder = os.path.join(root_folder, class_name)
    for image_file in os.listdir(class_folder):
        if image_file.endswith('.png'):  # Asegurarnos de solo procesar imágenes
            image_paths.append(os.path.join(class_folder, image_file))
            labels.append(label)  # Etiqueta de la clase (0 o 1)

# Convertir las listas a numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Dividir en train (80%) y test (20%) usando un 80/20
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

# Crear directorios para guardar las particiones de train y test
train_dir = 'eurosat/split/trainPI'
test_dir = 'eurosat/split/testPI'

# Crear las subcarpetas para las clases dentro de los directorios de train y test
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Mover las imágenes a sus respectivos directorios de train y test
for i, image_path in enumerate(X_train):
    label = y_train[i]
    class_name = classes[label]
    shutil.copy(image_path, os.path.join(train_dir, class_name, os.path.basename(image_path)))

for i, image_path in enumerate(X_test):
    label = y_test[i]
    class_name = classes[label]
    shutil.copy(image_path, os.path.join(test_dir, class_name, os.path.basename(image_path)))

print(f"Se han creado los conjuntos de train y test con {len(X_train)} imágenes para entrenamiento y {len(X_test)} para prueba.")





# %%

# %%
'''