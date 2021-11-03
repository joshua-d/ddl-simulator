import tensorflow as tf
import keras_model


# Dataset is collection of tuples (input, target)
mnist_dataset = keras_model.mnist_dataset().take(24)



dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# ds.shard(num_shards, shard_index)
shard_0 = dataset.shard(3, 0)
shard_1 = dataset.shard(3, 1)

print('Shard 0:')
for v in list(shard_0):
    print(v)

print('\nShard 1:')
for v in list(shard_1):
    print(v)


"""
Called on each worker

input_context = {
    num_input_pipelines: total num workers
    input_pipeline_id: THIS worker's id
}
"""

def dataset_fn(input_context):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)        # here each worker gets different shard
    return dataset

"""
Ex.
4 Workers

Worker 0:
dataset.shard(4, 0)

Worker 1:
dataset.shard(4, 1)

...

"""


def ex_dataset_fn(input_pipeline_id):
    worker_0_share = 0.25
    worker_1_share = 0.75

    num_worker_0_samples = int(worker_0_share * len(dataset))
    num_worker_1_samples = int(worker_1_share * len(dataset))

    if input_pipeline_id == 0:
        return dataset.take(num_worker_0_samples)

    elif input_pipeline_id == 1:
        return dataset.skip(num_worker_0_samples).take(num_worker_1_samples)


worker_0_ds = ex_dataset_fn(input_pipeline_id=0)
worker_1_ds = ex_dataset_fn(input_pipeline_id=1)

print('\nWorker 0 samples:')
print(list(worker_0_ds.as_numpy_iterator()))
print('\nWorker 1 samples:')
print(list(worker_1_ds.as_numpy_iterator()))


"""
If, given the worker's ID,
the worker's desired dataset may be obtained using Python code, 
then TF's code does not need to be modified
"""