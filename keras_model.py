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


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int64)
    return (x_test[0:num_samples], y_test[0:num_samples])


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='sigmoid', kernel_initializer=initializers.RandomNormal(), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(64, activation='sigmoid', kernel_initializer=initializers.RandomNormal(), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(10, kernel_initializer=initializers.RandomNormal(), bias_initializer=initializers.Zeros())
    ])

    return model


def build_model_with_seed(seed):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='sigmoid', kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(64, activation='sigmoid', kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(10, kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros())
    ])

    return model