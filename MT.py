# %%

import tensorflow as tf
from tensorflow.keras import layers, models

# U-Net para predecir la IR
def simple_unet(input_shape=(64, 64, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(pool2)
    conv3 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up1)
    conv3 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv3)

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    conv4 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)
    conv4 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv4)

    # Output layer (IR predicha)
    ir_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="ir_output")(conv4)

    return models.Model(inputs=inputs, outputs=ir_output)

# Clasificador binario
def create_simple_classifier(input_shape=(128, 64, 3)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    classification_output = layers.Dense(1, activation='sigmoid', name="classification_output")(x)

    return models.Model(inputs=inputs, outputs=classification_output)

# %%

from tensorflow.keras import layers, models

# Crear modelos individuales
unet_model = simple_unet(input_shape=(64, 64, 3))
classifier_model = create_simple_classifier(input_shape=(128, 64, 3))  # Concatenated RGB + IR

# Entrada principal (imagen RGB)
rgb_input = layers.Input(shape=(64, 64, 3))

# Salida de la U-Net (IR predicha)
ir_predicted = unet_model(rgb_input)  # Salida (64, 64, 1)

# Concatenar RGB con IR predicha en el eje vertical (espacial)
# Expandir el IR predicho a 3 canales para que coincida con RGB
ir_predicted_3c = layers.Concatenate(axis=-1)([ir_predicted] * 3)  # (64, 64, 3)

# Concatenar RGB e IR verticalmente
concatenated = layers.Concatenate(axis=1)([rgb_input, ir_predicted_3c])  # Salida (128, 64, 3)

# Clasificación binaria
classification_output = classifier_model(concatenated)

# Crear modelo combinado
# Crear modelo combinado con nombres explícitos para las salidas
multi_task_model = models.Model(
    inputs=rgb_input, 
    outputs={
        "ir_output": ir_predicted,                # Nombre explícito para salida IR
        "classification_output": classification_output  # Nombre explícito para salida de clasificación
    }
)


# %%

multi_task_model.compile(
    optimizer='adam',
    loss={
        "ir_output": "mean_squared_error",  # Reconstrucción IR
        "classification_output": "binary_crossentropy",  # Clasificación
    }
)

# %%

# Entrenamiento
history = multi_task_model.fit(
    train_rgb,  # Input: imágenes RGB
    {
        "ir_output": train_nir,  # Salida deseada para la U-Net
        "classification_output": train_labels,  # Etiquetas binarias para la clasificación
    },
    validation_data=(
        val_rgb,
        {
            "ir_output": val_nir,
            "classification_output": val_labels,
        }
    ),
    epochs=1,
    batch_size=500,
)

# %%
