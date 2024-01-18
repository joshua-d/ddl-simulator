import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers


# TODO not sure what these should be
batch_size = 64
learning_rate = 0.0001

optimizer_constructor = tf.keras.optimizers.Adam
loss_constructor = tf.keras.losses.SparseCategoricalCrossentropy
train_acc_metric_constructor = tf.keras.metrics.SparseCategoricalAccuracy

model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


def train_dataset():
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  return x_train, y_train


def test_dataset():
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int64)
    return x_test, y_test


def build_model_with_seed(seed):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed+1), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed+2), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed+3), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed+4), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(seed=seed+5), bias_initializer=initializers.Zeros()),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializers.RandomNormal(seed=seed+6), bias_initializer=initializers.Zeros())
    ])

    return model


# In dataset-rework, this just gives the master dataset which is automatically "sharded" by thread-safe DatasetIterator
def dataset_fn():
    x_train, y_train = train_dataset()
    mnist_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
    dataset = mnist_dataset.shuffle(len(mnist_dataset), seed=model_seed, reshuffle_each_iteration=False)
    return dataset


def model_builder():
    model = build_model_with_seed(model_seed)
    
    p_idx = 0
    params = {}

    for param in model.trainable_variables:
        params[p_idx] = param
        p_idx += 1

        train_acc_metric = train_acc_metric_constructor()

    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = loss_constructor(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        train_acc_metric.update_state(batch_targets, predictions)
        
        return grads_list

    def build_optimizer(learning_rate):
        return optimizer_constructor(learning_rate=learning_rate)

    return model, params, forward_pass, build_optimizer, loss_constructor(), train_acc_metric, batch_size, learning_rate
