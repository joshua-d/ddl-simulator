import tensorflow as tf
import multiprocessing
import keras_model
import time
import datetime
import multiprocessing



learning_rate = 0.1

num_epoches = 10
global_batch_size = 10
num_samples = 100

def dataset_fn(input_context):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.take(num_samples).shuffle(num_samples).repeat(num_epoches)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

steps_per_epoch = int(num_samples / global_batch_size)
accuracy_threshold = 0.9

model_seed = int(time.time())


# Cluster 1

# Set up cluster

def cluster1():

    cluster_dict_1 = {
        'worker': ['localhost:12345'],
        'ps': ['localhost:23456'],
        'chief': ['localhost:56789']
    }

    num_workers = 1
    worker_config_1 = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config_1.inter_op_parallelism_threads = num_workers + 1


    cluster_spec_1 = tf.train.ClusterSpec(cluster_dict_1)

    cluster_resolver_1 = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec_1, rpc_layer="grpc")


    # Create in-process servers

    # PS 1
    tf.distribute.Server(
            cluster_spec_1,
            job_name="ps",
            task_index=0,
            protocol="grpc")

    # Worker 1
    w1 = tf.distribute.Server(
            cluster_spec_1,
            job_name="worker",
            task_index=0,
            config=worker_config_1,
            protocol="grpc")


    strategy_1 = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver_1)


    with strategy_1.scope():
        multi_worker_model_1 = keras_model.build_model_with_seed(model_seed)
        optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        train_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_1')


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
            train_accuracy_1.update_state(batch_targets, predictions)
            return loss

        strategy_1.run(step_fn, args=(next(iterator),))
        return multi_worker_model_1.trainable_variables


    @tf.function
    def per_worker_dataset_fn_1():
        return strategy_1.distribute_datasets_from_function(dataset_fn)


    coordinator_1 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_1)

    per_worker_dataset_1 = coordinator_1.create_per_worker_dataset(per_worker_dataset_fn_1)
    per_worker_iterator_1 = iter(per_worker_dataset_1)




# Cluster 2

# Set up cluster

def cluster2():

    cluster_dict_2 = {
        'worker': ['localhost:34567'],
        'ps': ['localhost:45678'],
        'chief': ['localhost:56710']
    }

    num_workers = 1
    worker_config_2 = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config_2.inter_op_parallelism_threads = num_workers + 1


    cluster_spec_2 = tf.train.ClusterSpec(cluster_dict_2)

    cluster_resolver_2 = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec_2, rpc_layer="grpc")


    # Create in-process servers

    # PS 1
    tf.distribute.Server(
            cluster_spec_2,
            job_name="ps",
            task_index=0,
            protocol="grpc")

    # Worker 1
    w1 = tf.distribute.Server(
            cluster_spec_2,
            job_name="worker",
            task_index=0,
            config=worker_config_2,
            protocol="grpc")


    strategy_2 = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver_2)


    with strategy_2.scope():
        multi_worker_model_2 = keras_model.build_model_with_seed(model_seed)
        optimizer_2 = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        train_accuracy_2 = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy_2')


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
            train_accuracy_2.update_state(batch_targets, predictions)
            return loss

        strategy_2.run(step_fn, args=(next(iterator),))
        return multi_worker_model_2.trainable_variables


    @tf.function
    def per_worker_dataset_fn_2():
        return strategy_2.distribute_datasets_from_function(dataset_fn)


    coordinator_2 = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy_2)

    per_worker_dataset_2 = coordinator_2.create_per_worker_dataset(per_worker_dataset_fn_2)
    per_worker_iterator_2 = iter(per_worker_dataset_2)


for i in range(num_epoches):
    train_accuracy_1.reset_states()
    for _ in range(steps_per_epoch):
        w1_step = coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
    coordinator_1.join()
    print ("Finished epoch %d, accuracy is %f." % (i+1, train_accuracy_1.result().numpy()))


for i in range(num_epoches):
    train_accuracy_2.reset_states()
    for _ in range(steps_per_epoch):
        w2_step = coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,))
    coordinator_2.join()
    print ("2 Finished epoch %d, accuracy is %f." % (i+1, train_accuracy_2.result().numpy()))




t1 = threading.Thread(target=cluster1)
t2 = threading.Thread(target=cluster2)

cluster1()
cluster2()

# Top level model

# top_level_model = keras_model.build_model_with_seed(model_seed)

# top_K1 = top_level_model.layers[1].kernel
# top_B1 = top_level_model.layers[1].bias
# top_K2 = top_level_model.layers[2].kernel
# top_B2 = top_level_model.layers[2].bias

# w1_K1 = multi_worker_model_1.layers[1].kernel
# w1_B1 = multi_worker_model_1.layers[1].bias
# w1_K2 = multi_worker_model_1.layers[2].kernel
# w1_B2 = multi_worker_model_1.layers[2].bias

# w2_K1 = multi_worker_model_2.layers[1].kernel
# w2_B1 = multi_worker_model_2.layers[1].bias
# w2_K2 = multi_worker_model_2.layers[2].kernel
# w2_B2 = multi_worker_model_2.layers[2].bias


# def aggregate_top(w_K1, w_B1, w_K2, w_B2):
#     K1_val = (top_K1.value() + w_K1.value()) / 2
#     B1_val = (top_B1.value() + w_B1.value()) / 2
#     K2_val = (top_K2.value() + w_K2.value()) / 2
#     B2_val = (top_B2.value() + w_B2.value()) / 2

#     top_K1.assign(K1_val)
#     top_B1.assign(B1_val)
#     top_K2.assign(K2_val)
#     top_B2.assign(B2_val)

#     w_K1.assign(K1_val)
#     w_B1.assign(B1_val)
#     w_K2.assign(K2_val)
#     w_B2.assign(B2_val)


# def train():

#     num_complete_steps = 0

#     print('Beginning training')
#     start_time = time.time()

#     for i in range(num_epoches):
#         # train_accuracy_1.reset_states()

#         w1_step = coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
#         w2_step = coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,))

#         num_steps_this_epoch = 2

#         w1_done = w2_done = False

#         while not w1_done or not w2_done:
#             if not w1_done and w1_step._status.value[0] == 'R':  # w1 step complete
#                 # aggregate_top(w1_K1, w1_B1, w1_K2, w1_B2)

#                 v = w1_step.fetch()

#                 if num_steps_this_epoch == steps_per_epoch:
#                     w1_done = True
#                 else:
#                     w1_step = coordinator_1.schedule(train_step_1, args=(per_worker_iterator_1,))
#                     num_steps_this_epoch += 1
#                     num_complete_steps += 1

#             if not w2_done and w2_step._status.value[0] == 'R':  # w2 step complete
#                 # aggregate_top(w2_K1, w2_B1, w2_K2, w2_B2)

#                 v2 = w2_step.fetch()

#                 if num_steps_this_epoch == steps_per_epoch:
#                     w2_done = True
#                 else:
#                     w2_step = coordinator_2.schedule(train_step_2, args=(per_worker_iterator_2,)) 
#                     num_steps_this_epoch += 1
#                     num_complete_steps += 1

#         # print ("Finished epoch %d, accuracy is %f." % (i+1, train_accuracy_1.result().numpy()))
#         print('finished epoch ' + str(i+1))

                

        

        

# train()