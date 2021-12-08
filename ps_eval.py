import tensorflow as tf
import multiprocessing
import keras_model
import time
import datetime



learning_rate = 0.1

num_epoches = 20
global_batch_size = 10

num_train_samples = 5000
num_test_samples = 5000

def dataset_fn(input_context):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.take(num_train_samples).shuffle(num_train_samples).repeat(num_epoches)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

steps_per_epoch = int(num_train_samples / global_batch_size)
accuracy_threshold = 0.9

model_seed = int(time.time())



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
    multi_worker_model = keras_model.build_model_with_seed(model_seed)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


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

    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)



def train():

    x_test, y_test = keras_model.test_dataset(num_test_samples)
    accuracies = []

    print('Beginning training')
    start_time = time.time()

    for i in range(num_epoches):
        epoch = i+1

        for _ in range(steps_per_epoch):
            coordinator.schedule(train_step, args=(per_worker_iterator,))

        coordinator.join()
        print('Finished epoch %d' % epoch)

        predictions = multi_worker_model.predict(x_test)

        num_correct = 0
        for prediction, target in zip(predictions, y_test):
            answer = 0
            answer_val = prediction[0]
            for poss_ans_ind in range(len(prediction)):
                if prediction[poss_ans_ind] > answer_val:
                    answer = poss_ans_ind
                    answer_val = prediction[poss_ans_ind]
            if answer == target:
                num_correct += 1

        test_accuracy = float(num_correct) / num_test_samples
        print('Test accuracy: %f' % test_accuracy)

        accuracies.append(test_accuracy)


    time_elapsed = time.time() - start_time
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    with open('eval_logs/ps_eval_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()


train()