import tensorflow as tf
import keras_model
import time
import datetime
import threading
import json
import sys

from Cluster import Cluster


model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


config_file_path = "config.json"

def load_config():
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()
    return config


# cross worker data sharding does happen here
def dataset_fn(worker_id, num_train_samples):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shuffle(len(dataset), seed=(model_seed + worker_id)).take(num_train_samples)

    return dataset




def model_builder():
    model = keras_model.build_model_with_seed(model_seed)
    params = {
        'K1': model.layers[1].kernel,
        'B1': model.layers[1].bias,
        'K2': model.layers[2].kernel,
        'B2': model.layers[2].bias,
        # 'K3': model.layers[3].kernel,
        # 'B3': model.layers[3].bias
    }
    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        gradients = {
            'K1': grads_list[0],
            'B1': grads_list[1],
            'K2': grads_list[2],
            'B2': grads_list[3],
            # 'K3': grads_list[4],
            # 'B3': grads_list[5]
        }
        return gradients

    return model, params, forward_pass



def main():
    config = load_config()

    if len(sys.argv) > 1 and sys.argv[1] == 's':
        config['training_style'] = 'sync'
        print('******* SYNC TRAINING ********')

    cluster = Cluster(model_builder, dataset_fn, config)

    cluster.start()
    

main()