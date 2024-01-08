import tensorflow as tf
from threading import Lock

"""
This is a utility have an infinite iterator over a finite dataset,
which is automatically shuffled after each iteration.

This is possible with shuffle and repeat, that may take up a lot of memory when repeating
the dataset for many epochs
"""

# TODO consider generating a bunch in advance

# Original dataset is undisturbed - can create many DatasetIterators using one dataset - this feature is not currently used however
class DatasetIterator:

    def __init__(self, dataset, batch_size, reshuffle_each_iteration=True, initial_shuffle_seed=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.shuffle_seed = initial_shuffle_seed

        self.batched_dataset = dataset.batch(batch_size)
        self.batch_idx = 0

        self.iterator = iter(self.batched_dataset)
        

    def __next__(self):

        if self.batch_idx == len(self.batched_dataset):
            # Reached end of dataset
            if self.reshuffle_each_iteration:
                next_dataset = self.dataset.shuffle(1024, seed=self.shuffle_seed)
                self.shuffle_seed += 1
                self.batched_dataset = next_dataset.batch(self.batch_size)
            self.batch_idx = 0
            self.iterator = iter(self.batched_dataset)

        self.batch_idx += 1
        return next(self.iterator)
