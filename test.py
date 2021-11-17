import tensorflow as tf
import keras_model
import Worker, Cluster, ParameterServer


ds = keras_model.mnist_dataset()
ds = ds.batch(10)

it = iter(ds)

print(next(it))
print(next(it))