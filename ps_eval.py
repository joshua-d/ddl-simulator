import tensorflow as tf
import multiprocessing
import keras_model


# tf.debugging.set_log_device_placement(True)


# Set up cluster

cluster_dict = {
    'worker': ['localhost:12345', 'localhost:23456'],
    'ps': ['localhost:34567']
}

num_workers = 2
worker_config = tf.compat.v1.ConfigProto()
if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1


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


with strategy.scope():
    multi_worker_model = keras_model.build_model()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')


@tf.function
def train_step(iterator):

    def step_fn(inputs):
        batch_inputs, batch_targets = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model(batch_inputs, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)
            loss = tf.nn.compute_average_loss(
                per_batch_loss, global_batch_size=global_batch_size)

        grads = tape.gradient(loss, multi_worker_model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, multi_worker_model.trainable_variables))
        train_accuracy.update_state(batch_targets, predictions)
        return loss

    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


num_epoches = 5
global_batch_size = 10
num_samples = 10000

def dataset_fn(input_context):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.take(num_samples).shuffle(num_samples).repeat(num_epoches)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)



steps_per_epoch = int(num_samples / global_batch_size)

accuracy_threshold = 0.9


def train():

    incomplete_steps = []
    num_complete_steps = 0

    for i in range(num_epoches):
        train_accuracy.reset_states()
        for _ in range(steps_per_epoch):
            step = coordinator.schedule(train_step, args=(per_worker_iterator,))
            incomplete_steps.append(step)

        while len(incomplete_steps) != 0:
            for step in incomplete_steps:
                if step._status.value == 'READY':
                    incomplete_steps.remove(step)
                    num_complete_steps += 1
                    if train_accuracy.result().numpy() >= accuracy_threshold:
                        print('Accuracy threshold reached, %d steps' % num_complete_steps)
                        return

        print ("Finished epoch %d, accuracy is %f." % (i+1, train_accuracy.result().numpy()))

train()