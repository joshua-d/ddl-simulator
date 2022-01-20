import tensorflow as tf

"""
This is a utility have an infinite iterator over a finite dataset,
which is automatically shuffled after each iteration.

This is possible with shuffle and repeat, that may take up a lot of memory when repeating
the dataset for many epochs
"""

class DatasetIterator:

    def __init__(self, dataset, batch_size, reshuffle_each_iteration=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.reshuffle_each_iteration = reshuffle_each_iteration

        self.iterator = iter(dataset.batch(batch_size))
        

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            if self.reshuffle_each_iteration:
                next_dataset = self.dataset.take(len(self.dataset))
                next_dataset = next_dataset.shuffle(len(next_dataset))
                self.iterator = iter(next_dataset.batch(self.batch_size))
            else:
                self.iterator = iter(self.dataset.batch(self.batch_size))
            
            return next(self.iterator)

            