import tensorflow as tf



W1 = tf.Variable(
			tf.constant([
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2]
            ]),
			name='W1')


inp = tf.constant([
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]
                ])

print(W1)
print(inp)

m = tf.matmul(inp, W1)

print(m)

# each column in the weight matrix is a neuron, the values are its weights
# eat ass