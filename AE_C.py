# %%

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np


# %%

#-------------------------------------
#UNET SIMPLE FUNCTION
#-------------------------------------

def simple_unet(input_shape=(64, 64, 3)):
    inputs = layers.Input(input_shape)

    # Encoder: 2 capas de Conv2D
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # 2ª capa en Encoder
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(pool2)
    conv3 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up1)

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    conv4 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)

    # Output layer (un solo canal para la imagen IR)
    output = layers.Conv2D(1, (1, 1), activation="sigmoid")(conv4)  # Imagen IR simulada en escala de grises

    # Crear el modelo
    model = models.Model(inputs=inputs, outputs=output)

    # Compilar el modelo
    

    return model


# %%

#-------------------------------------
#LOAD TRAIN, VAL, TEST
#-------------------------------------

var = ['Highway', 'River']

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

# Load train data with labels
input_size = (64, 64)
train_rgb, train_nir, train_labels = load_image_pairs_with_labels('eurosat/split/train/', var, input_size=input_size)

# Load test data with labels
test_rgb, test_nir, test_labels = load_image_pairs_with_labels('eurosat/split/test/', var, input_size=input_size)

# Split train into train and validation
train_rgb, val_rgb, train_nir, val_nir, train_labels, val_labels = train_test_split(
    train_rgb, train_nir, train_labels, test_size=0.2, random_state=42
)

# Display the sizes of the splits
print(f"Train set size: {train_rgb.shape[0]} samples")
print(f"Validation set size: {val_rgb.shape[0]} samples")
print(f"Test set size: {test_rgb.shape[0]} samples")



# %%

#----------------------------
#TRAINING U-NET
#----------------------------

model = simple_unet(input_shape=(64, 64, 3))
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])


history = model.fit(
    train_rgb, train_nir,
    validation_data=(val_rgb, val_nir),
    epochs=2,
    batch_size=500
)


test_loss, test_mae = model.evaluate(test_rgb, test_nir)
print(f"Test Loss (MSE): {test_loss}, Test MAE: {test_mae}")

# %%

#---------------------------
#CONCAT RGB + PREDICTED IR
#---------------------------

# Predecir imágenes NIR
pre_train = model.predict(train_rgb)  # Predicciones NIR para entrenamiento
pre_val = model.predict(val_rgb)      # Predicciones NIR para validación
pre_test = model.predict(test_rgb)      # Predicciones NIR para validación

# Función para concatenar imágenes en una variable
def concatenate_images_in_memory(rgb_images, nir_predictions):
    concatenated_images = []

    for rgb, nir_pred in zip(rgb_images, nir_predictions):
        # Asegurar que NIR tenga 3 canales (opcional, si necesitas consistencia en forma)
        nir_pred_3channel = np.repeat(nir_pred, 3, axis=-1)  # Convertir de (64, 64, 1) a (64, 64, 3)
        
        # Concatenar verticalmente
        concatenated = np.vstack((rgb, nir_pred_3channel))
        
        # Agregar a la lista
        concatenated_images.append(concatenated)

    return np.array(concatenated_images)  # Convertir a numpy array para uso posterior

# Concatenar imágenes para entrenamiento y validación
train_concat = concatenate_images_in_memory(train_rgb, pre_train)
val_concat = concatenate_images_in_memory(val_rgb, pre_val)
test_concat = concatenate_images_in_memory(test_rgb, pre_test)

# Verificar formas
print("Forma de train_concat:", train_concat.shape)
print("Forma de val_concat:", val_concat.shape)

# Ahora puedes usar `train_concat` y `val_concat` como entrada para otro modelo
# %%


#-------------------
#CLASSIFICATION MODEL
#-------------------


# Parameters
batch_size = 32
n_iter = 1
ran = np.random.randint(1000, size=n_iter)
err = []

# Iterate with random seeds for train/validation splitting
for i in ran:
    # Create TensorFlow datasets for train and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_concat, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_concat, val_labels))

    # Batch and prefetch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define simple classification model
    def create_simple_model(input_shape=(128, 64, 3)):  # Updated for concatenated input
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        return model

    # Compile the model
    model = create_simple_model(input_shape=(128, 64, 3))  # Adjust input shape
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Add EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )

    # Train the model
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )

    # Predict on test set
    test_dataset = tf.data.Dataset.from_tensor_slices((test_concat, test_labels))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    predictions = np.round(np.ravel(model.predict(test_dataset)))

    # Calculate errors
    err.append(1 - accuracy_score(test_labels, predictions))

# Print errors
print(err)



# %%
'''
pre = model.predict(test_rgb)
j = 2

plt.imshow(pre[2])
plt.show()
plt.imshow(test_nir[2])
plt.show()

plt.imshow(test_rgb[2])
plt.show()
# %%
# %%
pre = model.predict(train_rgb)
j = 5

plt.imshow(pre[j])
plt.show()
plt.imshow(train_nir[j])
plt.show()

plt.imshow(train_rgb[j])
plt.show()
# %%
'''