import tensorflow as tf
import model

# each column in the weight matrix is a neuron, the values are its weights


ds = model.mnist_dataset()
ds = ds.batch(100)

mod = model.Model()

x, y = next(iter(ds))
mod.predict(x)