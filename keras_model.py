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
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_test[0:num_samples], y_test[0:num_samples]


def build_model_with_seed(seed):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None
    )

    # Add custom top layers for CIFAR-10 classification
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    # Combine the base model and custom top layers to create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    return model