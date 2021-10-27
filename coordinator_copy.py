import tensorflow as tf
import multiprocessing
import numpy as np
import keras_model
import model


# tf.debugging.set_log_device_placement(True)


# Set up cluster

cluster_dict = {
    'worker': ['localhost:12345', 'localhost:23456'],
    'ps': ['localhost:34567']
}

worker_config = tf.compat.v1.ConfigProto()
if multiprocessing.cpu_count() < 2 + 1:
    worker_config.inter_op_parallelism_threads = 2 + 1


cluster_spec = tf.train.ClusterSpec(cluster_dict)

cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")


# Create in-process servers

# PS 1
tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=0,
        protocol="grpc")

# Worker 1
w1 = tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=0,
        config=worker_config,
        protocol="grpc")

# Worker 2
tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=1,
        config=worker_config,
        protocol="grpc")


strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)


# Build model, set up optimizer (applies changes to model)

with strategy.scope():
  multi_worker_model = keras_model.build_model()
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')


# Custom training loop - training step function

@tf.function
def train_step(iterator):

    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model(x, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True,
              reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
            loss = tf.nn.compute_average_loss(
              per_batch_loss, global_batch_size=global_batch_size)

        print('trainable')
        print(multi_worker_model.trainable_variables)
        grads = tape.gradient(loss, multi_worker_model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, multi_worker_model.trainable_variables))
        train_accuracy.update_state(y, predictions)
        return loss

    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)



# Dataset functions (need more info on input context)

global_batch_size = 16

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shard(input_context.num_input_pipelines,      # User provides batching and sharding logic
                            input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset

@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)   # calls dataset_fn on each worker


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

# Here cluster coordinator sends dataset to each worker
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)


per_worker_iterator = iter(per_worker_dataset)

num_epoches = 10
steps_per_epoch = 5
for i in range(num_epoches):
    train_accuracy.reset_states()
    for _ in range(steps_per_epoch):
        coordinator.schedule(train_step, args=(per_worker_iterator,))
    # Wait at epoch boundaries.
    coordinator.join()
    print ("Finished epoch %d, accuracy is %f." % (i, train_accuracy.result().numpy()))