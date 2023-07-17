import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


def train_dataset():
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
  return train_dataset


def test_dataset(num_samples):
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_test[0:num_samples], y_test[0:num_samples])


def build_model_with_seed(seed):
    model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=10,
        classifier_activation="softmax"
    )

    return model