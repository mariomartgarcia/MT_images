import numpy as np
from sklearn.metrics import accuracy_score  
import tensorflow as tf


def compute_class_error_rates(test_labels, predictions, num_classes, class_error_list):
        """
        Compute error rate per class and append to the provided list.
        """
        for idx in range(num_classes):
                class_mask = (test_labels == idx)
                if np.sum(class_mask) > 0:
                        class_error = 1 - accuracy_score(test_labels[class_mask], predictions[class_mask])
                        class_error_list[idx].append(class_error)
        return class_error_list



# The temperature T is set to 1, if needed it can be changed as an argument of fcnn_multi.

def loss_GD_mc(l, num_classes):
    def loss(y_true, y_pred):
        y_tr = tf.keras.utils.to_categorical(y_true[:, 0], num_classes)            
        y_pr = tf.keras.utils.to_categorical(y_true[:, 1], num_classes)   

        d1 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        d2 = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_pr, y_pred)

        return tf.reduce_mean((1 - l) * d1 + l * d2)
    return loss


def loss_TPD_mc(beta, l, num_classes):
    def loss(y_true, y_pred):
        y_tr = tf.keras.utils.to_categorical(y_true[:, 0], num_classes)            
        y_prob = tf.keras.utils.to_categorical(y_true[:, 1], num_classes) 
        d = y_true[:, 2]
        
        bce_i = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_pred, y_prob)
        bce_r = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)

        return tf.reduce_mean((1-l)*(bce_r) + l*(tf.math.multiply(d,bce_i) - beta * tf.math.multiply(1-d, bce_i))) 
    return loss


def loss_MT_mc(num_classes):
    def loss(y_true, y_pred):
        pri = y_true[:, :, :, 0]
        y_tr = tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2))
        y_tr = tf.keras.utils.to_categorical(y_tr, num_classes)

        pi_pre = y_pred[:, :, :, 0]
        c_pre = tf.reduce_max(y_pred[:, :, :, 1:num_classes + 1], axis=(1, 2))
        sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
        temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

        ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, c_pre)

        l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
        l2 = (1/(tf.math.exp(temperature)))*ce  + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

        return tf.reduce_mean(l1 + l2)
    return loss

def loss_MT_PFD_mc(num_classes):
    def loss(y_true, y_pred):
        pri = y_true[:, :, :, 0]
        y_tr = tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2))
        y_tr = tf.keras.utils.to_categorical(y_tr, num_classes) 
        y_upper = tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2))
        y_upper = tf.keras.utils.to_categorical(y_upper, num_classes) 

        pi_pre = y_pred[:, :, :, 0]
        c_pre = tf.reduce_max(y_pred[:, :, :, 1:num_classes + 1], axis=(1, 2))
        sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
        temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

        l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
        bce = 0.5*tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_upper, c_pre) + 0.5*tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, c_pre) 
        l2 = (1/(tf.math.exp(temperature)))*(bce) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
        return tf.reduce_mean(l1 + l2)
    return loss

def loss_MT_TPD_mc(beta, num_classes):
    def loss(y_true, y_pred):
        pri = y_true[:, :, :, 0]
        y_tr = tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2))
        y_tr = tf.keras.utils.to_categorical(y_tr, num_classes) 
        y_upper = tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2))
        y_upper = tf.keras.utils.to_categorical(y_upper, num_classes) 
        d = tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2))
        
        pi_pre = y_pred[:, :, :, 0]
        c_pre = tf.reduce_max(y_pred[:, :, :, 1:num_classes + 1], axis=(1, 2))
        sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
        temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

        bce_i = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(c_pre, y_upper)
        bce_r = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, c_pre)

        l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
        l2 = (1/(tf.math.exp(temperature)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_i) - beta * tf.math.multiply(1-d, bce_i))) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
        return tf.reduce_mean(l1 + l2)
    return loss
