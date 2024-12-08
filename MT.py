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

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    conv4 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)

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




# %%

#STANDARD APPROACH


# Crear modelo combinado
# Crear modelo combinado con nombres explícitos para las salidas
multi_task_model = models.Model(
    inputs=rgb_input, 
    outputs=[ {
            "ir_output": ir_predicted,
            "classification_output": classification_output,
        }]

multi_task_model.compile(
    optimizer='adam',
    loss={
        "ir_output": "mean_squared_error",  # Reconstrucción IR
        "classification_output": "binary_crossentropy",  # Clasificación
    }
)

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




predictions = np.round(np.ravel(multi_task_model.predict(test_rgb)['classification_output']))
# Calculate errors
print(1 - accuracy_score(test_labels, predictions))

# %%


from tensorflow.keras import models
import tensorflow.keras.backend as K

# Definir función de pérdida personalizada
def custom_loss(y_true, y_pred):
    # Extraer las partes correspondientes de las salidas y las etiquetas
    ir_true, ir_pred = y_true[0], y_pred[0]  # Para IR
    classification_true, classification_pred = y_true[1], y_pred[1]  # Para clasificación

    # Calcular las pérdidas individuales
    mse_loss = K.mean(K.square(ir_true - ir_pred))  # Mean Squared Error
    bce_loss = K.binary_crossentropy(classification_true, classification_pred)  # Binary Crossentropy

    # Combinar las pérdidas
    return mse_loss + bce_loss

# Crear modelo combinado
multi_task_model = models.Model(
    inputs=rgb_input, 
    outputs=[
        ir_predicted,
        classification_output,
    ]
)

# Compilación del modelo
multi_task_model.compile(
    optimizer='adam',
    loss=custom_loss
)

# Entrenamiento
epochs = 1  # Modifica según sea necesario
batch_size = 500
history = multi_task_model.fit(
    train_rgb,  # Input: imágenes RGB
    [
        train_nir,  # Salida deseada para la U-Net
        train_labels,  # Etiquetas binarias para la clasificación
    ],
    validation_data=(
        val_rgb,
        [
            val_nir,
            val_labels,
        ]
    ),
    epochs=epochs,
    batch_size=batch_size,
)

# %%

#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
#NEW PART UNDER CONSTRUCTION 
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
def multi_task(y_true, y_pred):
    y_tr = max(y_true[:, 0])
    pri = y_true[:, 1]

    tf.print(y_true)
    tf.print(tf.shape(y_pred))
    c_pre, pi_pre  = y_pred[0], y_pred[1]

    #pi_pre = tf.reshape(pi_pre, [-1, 1]) 



    l1 = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction = 'none')(pi_pre, pri))
    l2 = tf.keras.losses.BinaryCrossentropy(reduction = 'none')(y_tr, c_pre)
    return tf.reduce_mean(l1 + l2)

#Joint model creation
# Definir función de pérdida personalizada
def custom_loss(y_true, y_pred):
    # Extraer las partes correspondientes de las salidas y las etiquetas
    ir_true, ir_pred = y_true[0], y_pred[0]  # Para IR
    classification_true, classification_pred = y_true[1], y_pred[1]  # Para clasificación

    # Calcular las pérdidas individuales
    mse_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction = 'none')(ir_true, ir_pred))  # Mean Squared Error
    bce_loss = K.binary_crossentropy(classification_true, classification_pred)  # Binary Crossentropy

    # Combinar las pérdidas
    return mse_loss + bce_loss

# Crear modelo combinado
# Expandir la forma del tensor classification_output
classification_output_expanded = layers.Reshape((1, 1, 1))(classification_output)
classification_output_upsampled = layers.UpSampling2D(size=(64, 64))(classification_output_expanded)  # Aumenta la resolución espacial
conc= layers.Concatenate(axis=-1)([ir_predicted, classification_output_upsampled])


multi_task_model = models.Model(
    inputs=rgb_input, 
    outputs=[conc
    ]
)

# Compilación del modelo
multi_task_model.compile(
    optimizer='adam',
    loss=multi_task
)

# Entrenamiento
epochs = 1  # Modifica según sea necesario
batch_size = 500
history = multi_task_model.fit(
    train_rgb,  # Input: imágenes RGB
    [
        train_nir,  # Salida deseada para la U-Net
        train_labels,  # Etiquetas binarias para la clasificación
    ],
    validation_data=(
        val_rgb,
        [
            val_nir,
            val_labels,
        ]
    ),
    epochs=epochs,
    batch_size=2,
)


'''
def multi_task(y_true, y_pred):
    y_tr = tf.reshape(y_true[:, 0], [-1,1])
    pri = tf.reshape(y_true[:, 1], [-1,1])


    #print(y_pred)
    c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=2, axis=1)
    c_pre = tf.reshape(c_pre, [-1, 1]) 
    pi_pre = tf.reshape(pi_pre, [-1, 1]) 
    #tf.print(c_pre)

    l1 = (1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
    l2 = (1/(tf.math.exp(sigma_T)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))
    return tf.reduce_mean(l1 + l2)
'''



# %%
def multi_task(y_true, y_pred):
    y_tr = y_true[:, 0]
    pri = y_true[:, 1]

    tf.print(y_tr)
    tf.print(pri)
    c_pre, pi_pre  = y_pred[0], y_pred[1]

    #pi_pre = tf.reshape(pi_pre, [-1, 1]) 


    l1 = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction = 'none')(pi_pre, pri))
    l2 = tf.keras.losses.BinaryCrossentropy(reduction = 'none')(y_tr, c_pre)
    return tf.reduce_mean(l1 + l2)


classification_output_expanded = layers.Reshape((1, 1, 1))(classification_output)
classification_output_upsampled = layers.UpSampling2D(size=(64, 64))(classification_output_expanded)  # Aumenta la resolución espacial
conc= layers.Concatenate(axis=-1)([ir_predicted, classification_output_upsampled])


multi_task_model = models.Model(
    inputs=rgb_input, 
    outputs=[conc
    ]
)




multi_task_model.compile(
    optimizer='adam',
    loss=multi_task
)



expanded_array = np.expand_dims(train_labels, axis=(-1, -2, -3))  # Forma [n, 1, 1, 1]
train_label_ex = np.tile(expanded_array, (1, 64, 64, 1)) 


expanded_array = np.expand_dims(val_labels, axis=(-1, -2, -3))  # Forma [n, 1, 1, 1]
val_label_ex = np.tile(expanded_array, (1, 64, 64, 1)) 

y_MT = np.column_stack([train_nir, train_label_ex])
y_MT_val = np.column_stack([val_nir, val_label_ex])



# Entrenamiento
history = multi_task_model.fit(
    train_rgb, y_MT,
    validation_data=(val_rgb, y_MT_val),
    epochs=1,
    batch_size=2,
)

# %%
