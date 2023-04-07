import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers, layers, models


def cifar10_dataset():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    return train_dataset


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_test = x_test / 255.0

    return (x_test[0:num_samples], y_test[0:num_samples])


def build_model_with_seed(seed):
    # TODO ADD KERNEL INITIALIZER SEED THING

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model