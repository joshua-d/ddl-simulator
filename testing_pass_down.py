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
tf.distribute.Server(
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
    v1 = tf.Variable(tf.constant([3, 3, 3]))


@tf.function
def train_step(iterator):
    return v1.value()


global_batch_size = 10


@tf.function
def per_worker_dataset_fn():
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(global_batch_size)
    return dataset


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)


for _ in range(10):
    rv = coordinator.schedule(train_step, args=(per_worker_iterator,))
    v = rv.fetch()
    print(v)
    v1.assign(tf.constant([4, 4, 4]))
    coordinator.join()

