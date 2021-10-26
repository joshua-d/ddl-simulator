import tensorflow as tf
import numpy as np


def xor_dataset():
    return np.array([
        ( [0, 0], [0] ),
        ( [0, 1], [1] ),
        ( [1, 0], [1] ),
        ( [1, 1], [0] )
    ], dtype=float)


class XORModel:

    def __init__(self):
        
        W1 = tf.constant([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        B1 = tf.constant([
            [0.1, 0.2]
        ])

        W2 = tf.constant([
            [0.5],
            [0.5]
        ])
        B2 = tf.constant([
            [0.1]
        ])

        self.params = [
            (W1, B1),
            (W2, B2)
        ]

    def predict(self, input):

        input_mat = tf.constant([input], dtype=float)
        for w, b in self.params:
            outputs = tf.math.add(tf.matmul(input_mat, w), b)
            input_mat = outputs

        print(outputs)

        

mod = XORModel()
mod.predict([0, 1])