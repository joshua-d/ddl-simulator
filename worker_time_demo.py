import multiprocessing
import os
import random
import portpicker
import tensorflow as tf
import time
from madb.vgg16_cifar10 import model_builder, dataset_fn


def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 10
NUM_PS = 1
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)


variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)



with strategy.scope():
  model, _, forward_pass, build_optimizer, _, accuracy, bs, lr = model_builder()
  optimizer = build_optimizer(lr)



@tf.function
def step_fn(iterator):

  def replica_fn(batch):
    start_time = time.perf_counter()
    forward_pass(batch, accuracy)
    end_time = time.perf_counter()
    return end_time - start_time

  batch = next(iterator)
  step_time = strategy.run(replica_fn, args=(batch,))
  return step_time


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)


@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)


num_epochs = 4
steps_per_epoch = 5
times = []
for i in range(num_epochs):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    step_time = coordinator.schedule(step_fn, args=(per_worker_iterator,))
    times.append(step_time)
  # Wait at epoch boundaries.
  coordinator.join()
  print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

for t in times:
  print(t.fetch())