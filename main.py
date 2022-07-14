import tensorflow as tf
import keras_model
import time
import datetime
import threading
import json
import sys
from multiprocessing import Process

from Cluster import Cluster
from DatasetIterator import DatasetIterator


model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


config_file_path = "config.json"

def load_config():
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()
    return config


mnist_dataset = keras_model.mnist_dataset()

def dataset_fn(worker_id, num_train_samples):
    master_dataset = mnist_dataset.shuffle(len(mnist_dataset), seed=model_seed).take(num_train_samples)

    # From here, data sharding is possible. Here, we don't shard - each worker has full dataset shuffled differently
    worker_dataset = master_dataset.shuffle(len(master_dataset), seed=(model_seed + worker_id))

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




def run_sim(config):
    cluster = Cluster(model_builder, dataset_fn, config)
    cluster.start()


def main():
    config = load_config()

    # 2 level tests


    # Bypass NI

    # S S
    for _ in range(20):
        p = Process(target=run_sim, args=(config,))
        p.start()
        p.join()

    # A S
    config['nodes'][0]['train_style'] = 'async'

    for _ in range(20):
        p = Process(target=run_sim, args=(config,))
        p.start()
        p.join()

    # S A
    config['nodes'][0]['train_style'] = 'sync'
    config['nodes'][1]['train_style'] = 'async'
    config['nodes'][2]['train_style'] = 'async'

    for _ in range(20):
        p = Process(target=run_sim, args=(config,))
        p.start()
        p.join()

    # A A
    config['nodes'][0]['train_style'] = 'async'

    for _ in range(20):
        p = Process(target=run_sim, args=(config,))
        p.start()
        p.join()

        

if __name__ == '__main__':
    main()