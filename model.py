import tensorflow as tf
import numpy as np


def mnist_dataset():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
    return train_dataset.take(1000)


class Model:

    def __init__(self):
        layer_lens = [784, 128, 10]

        W1 = tf.Variable(
            tf.random.normal([784, 128], stddev=5)
        )
        B1 = tf.Variable(
            tf.random.normal([1, 128])
        )

        W2 = tf.Variable(
            tf.random.normal([128, 10], stddev=5)
        )
        B2 = tf.Variable(
            tf.random.normal([1, 10])
        )

        self.params = [
            (W1, B1),
            (W2, B2)
        ]

        self.params_list = [W1, B1, W2, B2]

    def predict(self, input):
        input = tf.reshape(input, [1, 784])
        for w, b in self.params:
            output = tf.matmul(input, w) + b
            output = tf.nn.sigmoid(output)
            input = output

        return tf.reshape(output, (10,))

    def compute_loss(self, preds, targets):
        return tf.keras.losses.SparseCategoricalCrossentropy()(targets, preds)

    def step_fn(self, batch):
        with tf.GradientTape() as tape:
            preds = []
            batch_inputs, batch_targets = batch
            for input in batch_inputs:
                preds.append(self.predict(input))
            loss = self.compute_loss(preds, batch_targets)

        grads = tape.gradient(loss, self.params_list)
        tf.keras.optimizers.RMSprop(learning_rate=0.002).apply_gradients(zip(grads, self.params_list))
        return loss


mod = Model()
ds = mnist_dataset()
ds = ds.batch(100)

for i in range(60):
    for batch in ds:
        loss = mod.step_fn(batch)
    print('end of epoch ' + str(i))
    print('loss: ' + str(loss))


it = iter(ds.unbatch())
for i in range(5):
    smplx, smply = next(it)
    t = mod.predict(smplx)
    t = tf.math.round(t)
    print(t)
    print(smply)
    print()