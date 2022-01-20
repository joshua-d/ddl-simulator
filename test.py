from DatasetIterator import DatasetIterator
import keras_model
import tensorflow as tf


def dataset_fn(worker_id):
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataset = dataset.shuffle(10).take(10)
    return dataset, 2


ds, bat = dataset_fn(0)

di = DatasetIterator(ds, bat)

while True:
    print(next(di))
    print(len(di.dataset))
    input()

