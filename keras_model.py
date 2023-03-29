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

    # Define the DenseNet model architecture
    model = models.Sequential()

    # Add the initial convolutional layer
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Add four dense blocks with transition layers in between
    for i in range(4):
        # Dense block
        model.add(layers.Dense(32))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(128, (1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(32, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        
        # Transition layer
        model.add(layers.Conv2D(64, (1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

    # Add the final dense block
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.GlobalAveragePooling2D())

    # Add the output layer
    model.add(layers.Dense(10, activation='softmax'))

    return model