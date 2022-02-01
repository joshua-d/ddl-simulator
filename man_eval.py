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




# Set up clusters


cluster_dict_1 = {
    'worker': ['localhost:12345'],
    'ps': ['localhost:23456']
}

cluster_dict_2 = {
    'worker': ['localhost:11111'],
    'ps': ['localhost:33333']
}

num_workers = 1
worker_config_1 = tf.compat.v1.ConfigProto()
if multiprocessing.cpu_count() < num_workers + 1:
    worker_config_1.inter_op_parallelism_threads = num_workers + 1

worker_config_2 = tf.compat.v1.ConfigProto()
if multiprocessing.cpu_count() < num_workers + 1:
    worker_config_2.inter_op_parallelism_threads = num_workers + 1


cluster_spec_1 = tf.train.ClusterSpec(cluster_dict_1)
cluster_spec_2 = tf.train.ClusterSpec(cluster_dict_2)

cluster_resolver_1 = tf.distribute.cluster_resolver.SimpleClusterResolver(
    cluster_spec_1, rpc_layer="grpc")
cluster_resolver_2 = tf.distribute.cluster_resolver.SimpleClusterResolver(
    cluster_spec_2, rpc_layer="grpc")


# Create in-process servers

# PS 1
tf.distribute.Server(
        cluster_spec_1,
        job_name="ps",
        task_index=0,
        protocol="grpc")

# Worker 1
tf.distribute.Server(
        cluster_spec_1,
        job_name="worker",
        task_index=0,
        config=worker_config_1,
        protocol="grpc")

# PS 2
tf.distribute.Server(
        cluster_spec_2,
        job_name="ps",
        task_index=0,
        protocol="grpc")

# Worker 2
tf.distribute.Server(
        cluster_spec_2,
        job_name="worker",
        task_index=0,
        config=worker_config_2,
        protocol="grpc")


strategy_1 = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver_1)
strategy_2 = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver_2)

with strategy_1.scope():
    multi_worker_model_1 = keras_model.build_model_with_seed(model_seed)
    optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

with strategy_2.scope():
    multi_worker_model_2 = keras_model.build_model_with_seed(model_seed)
    optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


@tf.function
def train_step_1(iterator):

    def step_fn(inputs):
        batch_inputs, batch_targets = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model_1(batch_inputs, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)
            loss = tf.nn.compute_average_loss(
                per_batch_loss, global_batch_size=global_batch_size)

        grads = tape.gradient(loss, multi_worker_model_1.trainable_variables)
        optimizer_1.apply_gradients(
            zip(grads, multi_worker_model_1.trainable_variables))
        return loss

    per_replica_losses = strategy_1.run(step_fn, args=(next(iterator),))
    return strategy_1.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def train_step_2(iterator):

    def step_fn(inputs):
        batch_inputs, batch_targets = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model_2(batch_inputs, training=True)
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)
            loss = tf.nn.compute_average_loss(
                per_batch_loss, global_batch_size=global_batch_size)

        grads = tape.gradient(loss, multi_worker_model_2.trainable_variables)
        optimizer_2.apply_gradients(
            zip(grads, multi_worker_model_2.trainable_variables))
        return loss

    per_replica_losses = strategy_2.run(step_fn, args=(next(iterator),))
    return strategy_2.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def per_worker_dataset_fn_1():
    return strategy_1.distribute_datasets_from_function(dataset_fn)

@tf.function
def per_worker_dataset_fn_2():
    return strategy_2.distribute_datasets_from_function(dataset_fn)


coordinator_1 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_1)
per_worker_dataset_1 = coordinator_1.create_per_worker_dataset(per_worker_dataset_fn_1)
per_worker_iterator_1 = iter(per_worker_dataset_1)

coordinator_2 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_2)
per_worker_dataset_2 = coordinator_2.create_per_worker_dataset(per_worker_dataset_fn_2)
per_worker_iterator_2 = iter(per_worker_dataset_2)





# Top level model

top_level_model = keras_model.build_model_with_seed(model_seed)

top_K1 = top_level_model.layers[1].kernel
top_B1 = top_level_model.layers[1].bias
top_K2 = top_level_model.layers[2].kernel
top_B2 = top_level_model.layers[2].bias

w1_K1 = multi_worker_model_1.layers[1].kernel
w1_B1 = multi_worker_model_1.layers[1].bias
w1_K2 = multi_worker_model_1.layers[2].kernel
w1_B2 = multi_worker_model_1.layers[2].bias

w2_K1 = multi_worker_model_2.layers[1].kernel
w2_B1 = multi_worker_model_2.layers[1].bias
w2_K2 = multi_worker_model_2.layers[2].kernel
w2_B2 = multi_worker_model_2.layers[2].bias


def request_top(w_K1, w_B1, w_K2, w_B2):
    w_K1.assign(top_K1.value())
    w_B1.assign(top_B1.value())
    w_K2.assign(top_K2.value())
    w_B2.assign(top_B2.value())


def aggregate_top(w_K1, w_B1, w_K2, w_B2):
    top_K1.assign(w_K1.value())
    top_B1.assign(w_B1.value())
    top_K2.assign(w_K2.value())
    top_B2.assign(w_B2.value())


def train():

    x_test, y_test = keras_model.test_dataset(num_test_samples)
    accuracies = []

    print('Beginning training')
    start_time = time.time()

    for i in range(num_epoches):
        epoch = i+1

        w1_step = coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
        w2_step = coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,))

        num_steps_this_epoch = 2

        w1_done = w2_done = False

        while not w1_done or not w2_done:
            if not w1_done and w1_step._status.value[0] == 'R':  # w1 step complete
                aggregate_top(w1_K1, w1_B1, w1_K2, w1_B2)

                if num_steps_this_epoch == steps_per_epoch:
                    w1_done = True
                else:
                    request_top(w1_K1, w1_B1, w1_K2, w1_B2)
                    w1_step = coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
                    num_steps_this_epoch += 1

            if not w2_done and w2_step._status.value[0] == 'R':  # w2 step complete
                aggregate_top(w2_K1, w2_B1, w2_K2, w2_B2)

                if num_steps_this_epoch == steps_per_epoch:
                    w2_done = True
                else:
                    request_top(w2_K1, w2_B1, w2_K2, w2_B2)
                    w2_step = coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,)) 
                    num_steps_this_epoch += 1


        print("Finished epoch %d" % epoch)

        predictions = top_level_model.predict(x_test)

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

    with open('eval_logs/man_eval_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()

        

train()