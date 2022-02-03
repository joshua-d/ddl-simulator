import tensorflow as tf
import multiprocessing
import keras_model
import time
import datetime
import portpicker


num_workers = 2
num_ps = 1

learning_rate = 0.1

num_epoches = 2000
global_batch_size = 10

num_train_samples = 5000
num_test_samples = 5000

model_form = '784-128-10'

# num train samples per worker - workers may have a different set of train samples
def dataset_fn(input_context):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shuffle(num_train_samples*10).take(num_train_samples).shuffle(num_train_samples, reshuffle_each_iteration=True).repeat(num_epoches)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

steps_per_epoch = int(num_train_samples / global_batch_size)

model_seed = int(time.time())



# Set up cluster

worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

cluster_dict = {}
cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

cluster_spec = tf.train.ClusterSpec(cluster_dict)

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

    best_acc = 0
    best_acc_epoch = 0
    acc_delta = 0.005
    epochs_before_stop = 100
    epochs_under_delta = 0
    min_epochs = 200

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


        # Stop if accuracy has not risen 0.0002 above best acc in 200 epochs - 1 sample in 200 epochs

        if epoch > min_epochs: 
            if test_accuracy > best_acc and test_accuracy - best_acc > acc_delta:
                best_acc = test_accuracy
                best_acc_epoch = epoch
                epochs_under_delta = 0
            else:
                epochs_under_delta += 1

            if epochs_under_delta >= epochs_before_stop:
                break


    time_elapsed = time.time() - start_time
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    with open('eval_logs/ps_eval_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('%d workers, %d ps\n' % (num_workers, num_ps))
        outfile.write(model_form + '\n')
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        outfile.write('%d epochs before stop, %f accuracy delta, %d min epochs\n' % (epochs_before_stop, acc_delta, min_epochs))
        outfile.write('%d epochs, best accuracy: %f, epoch: %d\n\n' % (epoch, best_acc, best_acc_epoch))
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()


train()