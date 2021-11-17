import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


def mnist_dataset():
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
  return train_dataset


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='sigmoid', kernel_initializer=initializers.RandomNormal(seed=1), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(10, kernel_initializer=initializers.RandomNormal(seed=2), bias_initializer=initializers.Zeros())
    ])

    return model