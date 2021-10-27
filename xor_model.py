import tensorflow as tf


def xor_dataset():
    return (
        (tf.constant([[0.0, 0.0]]), tf.constant([[0.0]])),
        (tf.constant([[0.0, 1.0]]), tf.constant([[1.0]])),
        (tf.constant([[1.0, 0.0]]), tf.constant([[1.0]])),
        (tf.constant([[1.0, 1.0]]), tf.constant([[0.0]]))
    )


class XORModel:

    def __init__(self):
        
        W1 = tf.Variable(
            tf.random.normal([2, 2], stddev=5)
        )
        B1 = tf.Variable(
            tf.random.normal([1, 2])
        )

        W2 = tf.Variable(
            tf.random.normal([2, 1], stddev=5)
        )
        B2 = tf.Variable(
            tf.random.normal([1, 1])
        )

        self.params = [
            (W1, B1),
            (W2, B2)
        ]

        self.params_list = [W1, B1, W2, B2]

    def predict(self, input):
        for w, b in self.params:
            output = tf.matmul(input, w) + b
            output = tf.nn.sigmoid(output)
            input = output

        return output

    def compute_loss(self, preds, targets):
        return tf.keras.losses.BinaryCrossentropy()(targets, preds)

    def step_fn(self, batch):
        with tf.GradientTape() as tape:
            preds = []
            targets = []
            for datum in batch:
                input, target = datum
                preds.append(self.predict(input))
                targets.append(target)
            loss = self.compute_loss(preds, targets)

        grads = tape.gradient(loss, self.params_list)
        tf.keras.optimizers.SGD(learning_rate=0.5).apply_gradients(zip(grads, self.params_list))
        return loss

        

mod = XORModel()
dataset = xor_dataset()

print('Model:')
for p in mod.params_list:
    print(p)

loss = 1
while loss > 0.05:
    loss = mod.step_fn(dataset)
    print(loss)


print('Results: ')
for ex in xor_dataset():
    inputs, targets = ex
    print(mod.predict(inputs))

print('Model:')
for p in mod.params_list:
    print(p)


# mod = XORModel()
# print('Results: ')
# for ex in xor_dataset():
#     inputs, targets = ex
#     print(mod.predict(inputs))


