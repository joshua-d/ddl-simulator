import tensorflow as tf
import keras_model


ds = keras_model.mnist_dataset().take(20)

shuf1 = ds.shuffle(len(ds), seed=212)

shuf2 = ds.shuffle(len(ds), seed=212)

print(list(shuf1.as_numpy_iterator())[0])
print('\n\n\n')
print(list(shuf2.as_numpy_iterator())[0])