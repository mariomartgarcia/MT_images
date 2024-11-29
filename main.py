# %%
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import utils as utim
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np



#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/River')
#utim.extract_and_save_rgb_ir_eurosat('eurosat/EuroSATallBands/Forest')



var = ['AnnualCrop', 'PermanentCrop']
# %%

#----------------------------------------------------------------
#MODELO LOWER
#----------------------------------------------------------------

# Define the path to your dataset
dataset_dir = 'eurosat/fr_split/train/'

classes = [var[0] + 'RGB', var[1] + 'RGB']
# Collect file paths and labels
data = []
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            label = 0 if class_name == classes[0] else 1
            data.append((file_path, label))

# Create a DataFrame
df_train = pd.DataFrame(data, columns=['file_path', 'label'])
print(df_train.head())  # Verify data


# Define the path to your dataset
dataset_dir = 'eurosat/fr_split/test/'

# Collect file paths and labels
data = []
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            label = 0 if class_name == classes[0] else 1
            data.append((file_path, label))

# Create a DataFrame
df_test = pd.DataFrame(data, columns=['file_path', 'label'])
print(df_test.head())  # Verify data



# Función para cargar y preprocesar las imágenes
def load_and_preprocess_image(file_path, label, img_size=(64, 64)):
    img = tf.io.read_file(file_path)  # Leer la imagen desde el disco
    img = tf.image.decode_jpeg(img, channels=3)  # Decodificar la imagen JPEG
    img = tf.image.resize(img, img_size)  # Redimensionar la imagen
    img = img / 255.0  # Normalizar a rango [0, 1]
    return img, label

# Parámetros de entrenamiento
img_size = (64, 64)  # Tamaño de imagen
batch_size = 64

# Definir el modelo (por ejemplo, ResNet50)

#TEST DATA
test_dataset = tf.data.Dataset.from_tensor_slices((df_test['file_path'], df_test['label']))
test_dataset = test_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


n_iter  = 1
ran = np.random.randint(1000, size = n_iter)
err = []
for i in ran:
    train_files, val_files, train_labels, val_labels = train_test_split(
    df_train['file_path'], df_train['label'], test_size=0.2, stratify=df_train['label'], random_state=i)
        
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    # Map the loading and preprocessing function
    train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    '''
    model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(64, 64, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Clasificación binaria
    model = tf.keras.models.Model(inputs=model.input, outputs=x)
    '''
    def create_simple_model(input_shape=(64, 64, 3)):
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        return model

    model = create_simple_model(input_shape=(64, 64, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=5)



    # Train the model for the current fold with EarlyStopping
    model.fit(
        train_dataset, 
        epochs=10, 
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )

    pre = np.round(np.ravel(model.predict(test_dataset)))
    err.append(1- accuracy_score( df_test['label'], pre))

print(err)

# %%


#----------------------------------------------------------------
#MODELO UPPER
#----------------------------------------------------------------

# Define the path to your dataset
dataset_dir = 'eurosat/fr_split/trainPI/'
classes = [var[0] + 'PI', var[1] + 'PI']
# Collect file paths and labels
data = []
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            label = 0 if class_name == classes[0] else 1
            data.append((file_path, label))

# Create a DataFrame
df_train = pd.DataFrame(data, columns=['file_path', 'label'])
print(df_train.head())  # Verify data


# Define the path to your dataset
dataset_dir = 'eurosat/fr_split/testPI/'

# Collect file paths and labels
data = []
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            label = 0 if class_name == classes[0] else 1
            data.append((file_path, label))

# Create a DataFrame
df_test = pd.DataFrame(data, columns=['file_path', 'label'])
print(df_test.head())  # Verify data


img_size = (64, 128)  # Tamaño de imagen
# Función para cargar y preprocesar las imágenes
def load_and_preprocess_image(file_path, label, img_size=(64, 128)):
    img = tf.io.read_file(file_path)  # Leer la imagen desde el disco
    img = tf.image.decode_jpeg(img, channels=3)  # Decodificar la imagen JPEG
    img = tf.image.resize(img, img_size)  # Redimensionar la imagen
    img = img / 255.0  # Normalizar a rango [0, 1]
    return img, label

# Parámetros de entrenamiento
batch_size = 64

# Definir el modelo (por ejemplo, ResNet50)

#TEST DATA
test_dataset = tf.data.Dataset.from_tensor_slices((df_test['file_path'], df_test['label']))
test_dataset = test_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


n_iter  = 1
ran = np.random.randint(1000, size = n_iter)
err = []
for i in ran:
    train_files, val_files, train_labels, val_labels = train_test_split(
    df_train['file_path'], df_train['label'], test_size=0.2, stratify=df_train['label'], random_state=i)
        
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    # Map the loading and preprocessing function
    train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    '''
    model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(64, 64, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Clasificación binaria
    model = tf.keras.models.Model(inputs=model.input, outputs=x)
    '''
    def create_simple_model(input_shape=(64, 64, 3)):
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        return model

    model = create_simple_model(input_shape=(64, 128, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=5)



    # Train the model for the current fold with EarlyStopping
    model.fit(
        train_dataset, 
        epochs=10, 
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )

    pre = np.round(np.ravel(model.predict(test_dataset)))
    err.append(1- accuracy_score( df_test['label'], pre))

print(err)

# %%
