
import tensorflow as tf
from tensorflow.keras import layers, models


def fcnn(input_shape=(64, 64, 3)):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    return model


# U-Net para predecir la IR
def simple_unet_ini(input_shape=(64, 64, 3)):
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
    conv3_ex = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up1)  #Conv extra
    conv3 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv3_ex) 

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    conv4_ex = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)  #Conv extra
    conv4 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv4_ex)

    #Output layer (IR predicha)
    #Sigmoid activation. NIR is normalized between [0,1].
    ir_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="ir_output")(conv4)

    return models.Model(inputs=inputs, outputs=ir_output)


def simple_unet(input_shape=(64, 64, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    up1 = layers.Concatenate()([up1, conv2])
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up1)
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv4)

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv4)
    up2 = layers.Concatenate()([up2, conv1])
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv5)

    # Output layer (IR predicha)
    ir_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="ir_output")(conv5)

    return models.Model(inputs=inputs, outputs=ir_output)


class ExtLayer(layers.Layer):
        def __init__(self, initial_sigma=0.1):
            super(ExtLayer, self).__init__()
            # Initialize the trainable variable
            self.var = tf.Variable(initial_value=[[initial_sigma]], dtype=tf.float32, trainable=True)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]  # Get the current batch size
            # Tile sigma to match the batch size
            tensor = tf.tile(self.var, [batch_size, 1])
            return tensor

#Multi-task model

def MT(reg_sh = (64, 64, 3), pri_sh = (128, 64, 3)):
    unet_model = simple_unet(input_shape = reg_sh)
    classifier_model = fcnn(input_shape = pri_sh) 

    rgb_input = layers.Input(shape=reg_sh)
    ir_predicted = unet_model(rgb_input)  

    # Concatenate RGB with predicted IR along the vertical axis (spatial)
    # Expand the predicted IR to 3 channels to match RGB
    # Concatenate RGB and IR vertically
    ir_predicted_3c = layers.Concatenate(axis=-1)([ir_predicted] * 3)  # (64, 64, 3)
    concatenated = layers.Concatenate(axis=1)([rgb_input, ir_predicted_3c])  # (128, 64, 3)

    classification_output = classifier_model(concatenated)

    classification_output_expanded = layers.Reshape((1, 1, 1))(classification_output)
    classification_output_upsampled = layers.UpSampling2D(size=(64, 64))(classification_output_expanded) 


    #External parameters
    sigma_output = ExtLayer()(rgb_input)
    temp_output = ExtLayer()(rgb_input)


    sigma_out_expand = layers.Reshape((1, 1, 1))(sigma_output)
    sigma = layers.UpSampling2D(size=(64, 64))(sigma_out_expand) 

    temperature_out_expand = layers.Reshape((1, 1, 1))(temp_output)
    temperature = layers.UpSampling2D(size=(64, 64))(temperature_out_expand)


    conc= layers.Concatenate(axis=-1)([ir_predicted, classification_output_upsampled,
                                        sigma, temperature])

    multi_task_model = models.Model(inputs=rgb_input, outputs=[conc])
    
    return multi_task_model


def MT_band(reg_sh = (64, 64, 3), pri_sh = (64, 64, 4)):
    unet_model = simple_unet(input_shape = reg_sh)
    classifier_model = fcnn(input_shape = pri_sh) 

    rgb_input = layers.Input(shape=reg_sh)
    ir_predicted = unet_model(rgb_input)  

    concatenated = layers.Concatenate(axis=-1)([rgb_input, ir_predicted])

    classification_output = classifier_model(concatenated)

    classification_output_expanded = layers.Reshape((1, 1, 1))(classification_output)
    classification_output_upsampled = layers.UpSampling2D(size=(64, 64))(classification_output_expanded) 


    #External parameters
    sigma_output = ExtLayer()(rgb_input)
    temp_output = ExtLayer()(rgb_input)


    sigma_out_expand = layers.Reshape((1, 1, 1))(sigma_output)
    sigma = layers.UpSampling2D(size=(64, 64))(sigma_out_expand) 

    temperature_out_expand = layers.Reshape((1, 1, 1))(temp_output)
    temperature = layers.UpSampling2D(size=(64, 64))(temperature_out_expand)


    conc= layers.Concatenate(axis=-1)([ir_predicted, classification_output_upsampled,
                                        sigma, temperature])

    multi_task_model = models.Model(inputs=rgb_input, outputs=[conc])
    
    return multi_task_model



#PARA 256

def fcnn256(input_shape=(256, 256, 3)):
    model = tf.keras.Sequential([
        # Primera capa convolucional
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Reducción a 128x128x32

        # Segunda capa convolucional
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Reducción a 64x64x64

        # Tercera capa convolucional
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Reducción a 32x32x128

        # Cuarta capa convolucional
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Reducción a 16x16x256

        # Capa de Global Average Pooling
        tf.keras.layers.GlobalAveragePooling2D(),  # Reducción a 256

        # Capa densa
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria
    ])
    return model


# U-Net para predecir la IR
def simple_unet256(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)  # Reducción a 32x32x128

    conv4 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
    conv4 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)  # Reducción a 16x16x256

    # Decoder
    up1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(pool2)
    conv5_ex = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(up1)  #Conv extra
    conv5 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(conv5_ex) 

    up2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv5)
    conv6_ex = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(up2)  #Conv extra
    conv6 = layers.Conv2D(126, (3, 3), padding="same", activation="relu")(conv6_ex) 

    up3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv6)
    conv7_ex = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up3)  #Conv extra
    conv7 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv7_ex) 

    up4 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv7)
    conv8_ex = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up4)  #Conv extra
    conv8 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv8_ex)

    #Output layer (IR predicha)
    #Sigmoid activation. NIR is normalized between [0,1].
    ir_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="ir_output")(conv8)

    return models.Model(inputs=inputs, outputs=ir_output)