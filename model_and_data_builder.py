import tensorflow as tf
import keras_model


model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


mnist_dataset = keras_model.mnist_dataset()

def dataset_fn(num_workers, worker_idx, num_train_samples):
    master_dataset = mnist_dataset.shuffle(len(mnist_dataset), seed=model_seed, reshuffle_each_iteration=False).take(num_train_samples)

    worker_dataset = master_dataset.shard(num_shards=num_workers, index=worker_idx)

    return worker_dataset



def model_builder():
    model = keras_model.build_model_with_seed(model_seed)
    
    p_idx = 0
    params = {}

    for param in model.trainable_variables:
        params[p_idx] = param
        p_idx += 1

    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        
        return grads_list

    def build_optimizer(learning_rate):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return model, params, forward_pass, build_optimizer