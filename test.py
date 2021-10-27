import tensorflow as tf
import keras_model

# each column in the weight matrix is a neuron, the values are its weights


ds = keras_model.mnist_dataset()
ds = ds.batch(100)

mod = keras_model.build_model()

x, y = next(iter(ds))
mod.predict(x)