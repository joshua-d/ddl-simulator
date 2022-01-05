import tensorflow as tf
import keras_model
from Worker import Worker
from ParameterServer import ParameterServer
from Cluster import Cluster
import time
import datetime


learning_rate = 0.1

num_epoches = 10
global_batch_size = 10

num_train_samples = 5000
num_test_samples = 5000


# still no cross-worker data sharding, emulates manual-aggr-eval system
def dataset_fn():
    dataset = keras_model.mnist_dataset()
    dataset = dataset.take(num_train_samples).shuffle(num_train_samples).repeat(num_epoches)
    dataset = dataset.batch(global_batch_size)
    return dataset


steps_per_epoch = int(num_train_samples / global_batch_size)


def model_builder():
    model = keras_model.build_model()
    params = {
        'K1': model.layers[1].kernel,
        'B1': model.layers[1].bias,
        'K2': model.layers[2].kernel,
        'B2': model.layers[2].bias
    }
    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        gradients = {
            'K1': grads_list[0],
            'B1': grads_list[1],
            'K2': grads_list[2],
            'B2': grads_list[3]
        }
        return gradients

    return model, params, forward_pass


# this just used to create the vars on the parameter server
initial_model = keras_model.build_model()

params = {
    'K1': initial_model.layers[1].kernel,
    'B1': initial_model.layers[1].bias,
    'K2': initial_model.layers[2].kernel,
    'B2': initial_model.layers[2].bias
}

ps = ParameterServer(params, tf.keras.optimizers.RMSprop(learning_rate=learning_rate))


cl = Cluster()
cl.parameter_servers = {
    'ps1': ps
}


param_locations = {
    'ps1': ['K1', 'B1', 'K2', 'B2']
}


w1 = Worker(cl, model_builder, iter(dataset_fn()), param_locations)
w2 = Worker(cl, model_builder, iter(dataset_fn()), param_locations)


def eval_once():
    num_test_samples = 300
    x_test, y_test = keras_model.test_dataset(num_test_samples)
    predictions = initial_model.predict(x_test)

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




def train():

    x_test, y_test = keras_model.test_dataset(num_test_samples)
    accuracies = []

    print('Beginning training')
    start_time = time.time()

    for i in range(num_epoches):
        epoch = i+1

        # schedule steps - need threading
        for _ in range(int(steps_per_epoch / 2)):
            w1.train_step()
            w2.train_step()

        print('Finished epoch %d' % epoch)

        predictions = initial_model.predict(x_test)

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