import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


def train_dataset():
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 255.0
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  return x_train, y_train


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_test[0:num_samples], y_test[0:num_samples]


def build_model_with_seed(seed):
    vgg_base = tf.keras.applications.VGG16(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None
    )

    # Create a new model and add the VGG16 base
    model = tf.keras.models.Sequential()
    model.add(vgg_base)

    # Add custom top layers for CIFAR-10
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model