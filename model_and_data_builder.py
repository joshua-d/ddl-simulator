import tensorflow as tf
import keras_model


model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


# In dataset-rework, this just gives the master dataset which is automatically "sharded" by thread-safe DatasetIterator
def dataset_fn(num_train_samples):
    x_train, y_train = keras_model.train_dataset()
    cifar10_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
    dataset = cifar10_dataset.shuffle(len(cifar10_dataset), seed=model_seed, reshuffle_each_iteration=False).take(num_train_samples)
    return dataset


def model_builder():
    model = keras_model.build_model_with_seed(model_seed)
    
    p_idx = 0
    params = {}

    for param in model.trainable_variables:
        params[p_idx] = param
        p_idx += 1

    loss_type = tf.keras.losses.CategoricalCrossentropy()

    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = loss_type(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        
        return grads_list, loss

    def build_optimizer(learning_rate):
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6)

    return model, params, forward_pass, build_optimizer, loss_type
