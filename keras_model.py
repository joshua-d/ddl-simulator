import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


def mnist_dataset():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)

    x_train = tf.expand_dims(x_train, axis=3)
    x_train = tf.repeat(x_train, 3, axis=3)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    return train_dataset


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int64)

    x_test = tf.expand_dims(x_test, axis=3)
    x_test = tf.repeat(x_test, 3, axis=3)

    return (x_test[0:num_samples], y_test[0:num_samples])


# strides and sigmoid activation added according to wikipedia
def build_model_with_seed(seed):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(224, 224, input_shape=(28, 28, 3)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),

        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
        
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10)
    ])

    return model