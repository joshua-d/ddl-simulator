import tensorflow as tf
import keras_model

# each column in the weight matrix is a neuron, the values are its weights


model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

print(model2.layers[1].kernel.value())


v1 = tf.constant([[1, 2, 3],
                  [4, 5, 6]])
v2 = tf.constant([[2, 3, 4],
                  [5, 6, 7]])

r = (v1 + v2) / 2
print(r)