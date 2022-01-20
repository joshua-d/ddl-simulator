from glob import glob
import tensorflow as tf
import keras_model
import time
import datetime
import threading
import json

from Cluster import Cluster

from tts import tts


learning_rate = 0.1

num_epoches = 300
global_batch_size = 10

num_train_samples = 5000
num_test_samples = 5000


config = {}
config_file_path = "config.json"
def load_config():
    global config
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()



# still no cross-worker data sharding, emulates manual-aggr-eval system
def dataset_fn(worker_id):
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shuffle(num_train_samples*10).take(num_train_samples)
    batch_size = global_batch_size

    return dataset, batch_size


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



def train(cluster):

    x_test, y_test = keras_model.test_dataset(num_test_samples)
    accuracies = []

    print('Beginning training')
    start_time = time.time()

    best_acc = 0
    stop_counter = 0

    for i in range(num_epoches):
        epoch = i+1

        cluster.steps_completed = 0
        cluster.steps_scheduled = steps_per_epoch

        w1_thread = threading.Thread(target=cluster.workers[0].train, daemon=True)
        w2_thread = threading.Thread(target=cluster.workers[1].train, daemon=True)

        w1_thread.start()
        w2_thread.start()

        w1_thread.join()
        w2_thread.join()

        print('Finished epoch %d' % epoch)

        predictions = cluster.get_test_model().predict(x_test)

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

        if epoch % 10 == 0:
            tts('Eapock %d, ack %d' % (epoch, int(test_accuracy*100)))

        accuracies.append(test_accuracy)


        # Stop if it reaches 0.95, or drops 0.2 below the best acc and stays there for 10 epochs in a row

        if test_accuracy >= 0.95:
            break

        if test_accuracy > best_acc:
            best_acc = test_accuracy

        if best_acc - test_accuracy > 0.2:
            stop_counter += 1
        else:
            stop_counter = 0

        if stop_counter >= 10:
            break


    time_elapsed = time.time() - start_time
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    best_acc = 0
    for acc in accuracies:
        if acc > best_acc:
            best_acc = acc

    with open('eval_logs/custom_ps_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        outfile.write('%d epochs, best accuracy: %f\n\n' % (epoch, best_acc))
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()


def main():
    load_config()

    cluster = Cluster(model_builder, dataset_fn, config)

    train(cluster)
    

main()