import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


def cifar10_dataset():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    return train_dataset


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_test[0:num_samples], y_test[0:num_samples])


# strides and sigmoid activation added according to wikipedia
def build_model_with_seed(seed):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(32, 32, 3), padding="same", activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros()),

        tf.keras.layers.AveragePooling2D(strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(seed=(seed+1)), bias_initializer=initializers.Zeros()),

        tf.keras.layers.AveragePooling2D(strides=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120, activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(seed=(seed+2)), bias_initializer=initializers.Zeros()),

        tf.keras.layers.Dense(84, activation="sigmoid",
            kernel_initializer=initializers.RandomNormal(seed=(seed+3)), bias_initializer=initializers.Zeros()),
            
        tf.keras.layers.Dense(10,
            kernel_initializer=initializers.RandomNormal(seed=(seed+4)), bias_initializer=initializers.Zeros()) # softmax activation?
    ])

    return model