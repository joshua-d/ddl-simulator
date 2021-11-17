import tensorflow as tf
import multiprocessing
import keras_model
import os

# tf.debugging.set_log_device_placement(True)

# Set up clusters

cluster_1 = {
    'worker': ['localhost:12345'],
    'ps': ['localhost:34567']
}

cluster_2 = {
    'worker': ['localhost:11111'],
    'ps': ['localhost:33333']
}

cluster_1_spec = tf.train.ClusterSpec(cluster_1)
cluster_2_spec = tf.train.ClusterSpec(cluster_2)

cluster_1_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_1_spec, rpc_layer="grpc")

cluster_2_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_2_spec, rpc_layer="grpc")


# Create in-process servers

# Cluster 1
tf.distribute.Server(
        cluster_1,
        job_name="ps",
        task_index=0,
        protocol="grpc")

tf.distribute.Server(
        cluster_1,
        job_name="worker",
        task_index=0,
        protocol="grpc")

# Cluster 2
tf.distribute.Server(
        cluster_2,
        job_name="ps",
        task_index=0,
        protocol="grpc")

tf.distribute.Server(
        cluster_2,
        job_name="worker",
        task_index=0,
        protocol="grpc")


strategy_1 = tf.distribute.experimental.ParameterServerStrategy(cluster_1_resolver)
strategy_2 = tf.distribute.experimental.ParameterServerStrategy(cluster_2_resolver)



with strategy_1.scope():
    model_1 = keras_model.build_model()
    optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    train_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    
with strategy_2.scope():
    model_2 = keras_model.build_model()
    optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    train_accuracy_2 = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')


@tf.function
def train_step_1(iterator):

    def step_fn(inputs):
        batch_inputs, batch_targets = inputs
        with tf.GradientTape() as tape:
            predictions = model_1(batch_inputs, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)
            loss = tf.nn.compute_average_loss(
                per_batch_loss, global_batch_size=global_batch_size)

        grads = tape.gradient(loss, model_1.trainable_variables)
        optimizer_1.apply_gradients(
            zip(grads, model_1.trainable_variables))
        train_accuracy_1.update_state(batch_targets, predictions)
        return loss

    per_replica_losses = strategy_1.run(step_fn, args=(next(iterator),))  # calls step fn on each replica on this worker, each replica processes a different batch
    return strategy_1.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)  # sum of losses across replicas


@tf.function
def train_step_2(iterator):

    def step_fn(inputs):
        batch_inputs, batch_targets = inputs
        with tf.GradientTape() as tape:
            predictions = model_2(batch_inputs, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)
            loss = tf.nn.compute_average_loss(
                per_batch_loss, global_batch_size=global_batch_size)

        grads = tape.gradient(loss, model_2.trainable_variables)
        optimizer_2.apply_gradients(
            zip(grads, model_2.trainable_variables))
        train_accuracy_2.update_state(batch_targets, predictions)
        return loss

    per_replica_losses = strategy_2.run(step_fn, args=(next(iterator),))  # calls step fn on each replica on this worker, each replica processes a different batch
    return strategy_2.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)  # sum of losses across replicas


global_batch_size = 10

def dataset_fn_1(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shard(2,
                            0) # first shard of data
    dataset = dataset.batch(batch_size)
    return dataset

def dataset_fn_2(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shard(2,
                            1) # second shard of data
    dataset = dataset.batch(batch_size)
    return dataset

@tf.function
def per_worker_dataset_fn_1():
    return strategy_1.distribute_datasets_from_function(dataset_fn_1)

@tf.function
def per_worker_dataset_fn_2():
    return strategy_2.distribute_datasets_from_function(dataset_fn_2)


coordinator_1 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_1)
per_worker_dataset_1 = coordinator_1.create_per_worker_dataset(per_worker_dataset_fn_1)  # calls dataset_fn on each worker
per_worker_iterator_1 = iter(per_worker_dataset_1)

coordinator_2 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_2)
per_worker_dataset_2 = coordinator_2.create_per_worker_dataset(per_worker_dataset_fn_2)  # calls dataset_fn on each worker
per_worker_iterator_2 = iter(per_worker_dataset_2)



aggregated_model = keras_model.build_model()

K1 = aggregated_model.layers[1].kernel
B1 = aggregated_model.layers[1].bias
K2 = aggregated_model.layers[2].kernel
B2 = aggregated_model.layers[2].bias

aggregated_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

ds = keras_model.mnist_dataset().take(100)
ds = ds.batch(10)


num_epoches = 10
steps_per_epoch = 100

for i in range(num_epoches):
    train_accuracy_1.reset_states()
    train_accuracy_2.reset_states()

    for _ in range(steps_per_epoch):
        # Schedule a step on both workers
        coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
        coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,))
        
        # Wait for both to finish (synchronous - would need to implement a threading system for async)
        coordinator_1.join()
        coordinator_2.join()

        # Aggregate - average each model into existing top-level model 1 by 1
        K1_val = (K1.value() + model_1.layers[1].kernel.value()) / 2
        B1_val = (B1.value() + model_1.layers[1].bias.value()) / 2
        K2_val = (K2.value() + model_1.layers[2].kernel.value()) / 2
        B2_val = (B2.value() + model_1.layers[2].bias.value()) / 2

        K1.assign(K1_val)
        B1.assign(B1_val)
        K2.assign(K2_val)
        B2.assign(B2_val)

        K1_val = (K1.value() + model_2.layers[1].kernel.value()) / 2
        B1_val = (B1.value() + model_2.layers[1].bias.value()) / 2
        K2_val = (K2.value() + model_2.layers[2].kernel.value()) / 2
        B2_val = (B2.value() + model_2.layers[2].bias.value()) / 2

        K1.assign(K1_val)
        B1.assign(B1_val)
        K2.assign(K2_val)
        B2.assign(B2_val)

        
    # Wait at epoch boundaries.
    coordinator_1.join()
    coordinator_2.join()
    print("Finished epoch %d, accuracy 1 is %f." % (i, train_accuracy_1.result().numpy()))
    print("Finished epoch %d, accuracy 2 is %f." % (i, train_accuracy_2.result().numpy()))




aggregated_model.evaluate(ds)

b = next(iter(ds))
x, y = b

print('Correct outputs: ')
print(y)
print()

print('Model 1 outputs: ')
print(model_1(x))

print()

print('Correct outputs: ')
print(y)
print()

print('Aggregated model outputs:')
print(aggregated_model(x))
