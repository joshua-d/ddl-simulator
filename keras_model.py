import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
import tensorflow_datasets as tfds

vocab_size = 1000

# 25000 train, 25000 test

def imdb_dataset():
  dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
  return dataset['train']


def test_dataset(num_samples):
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    return dataset['test'].take(num_samples)


def build_model_with_seed(seed):
    train_dataset = imdb_dataset()
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
            kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros())),
        tf.keras.layers.Dense(64, activation='relu',
            kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(1,
            kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros())
    ])

    # So that the model weights will be built - could also specify input_shape
    model.predict(np.array(['sample text']))

    return model