import tensorflow as tf
import keras_model
import time
import datetime
import threading
import json

from Cluster import Cluster

from tts import tts


learning_rate = 0.1

# num_epoches = 600
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
        'B2': model.layers[2].bias,
        # 'K3': model.layers[3].kernel,
        # 'B3': model.layers[3].bias
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
            'B2': grads_list[3],
            # 'K3': grads_list[4],
            # 'B3': grads_list[5]
        }
        return gradients

    return model, params, forward_pass



def train(cluster):

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

    epoch = 0

    while True:
        epoch += 1

        cluster.steps_completed = 0
        cluster.steps_scheduled = steps_per_epoch

        w1_thread = threading.Thread(target=cluster.workers[0].train, daemon=True)
        w2_thread = threading.Thread(target=cluster.workers[1].train, daemon=True)
        w3_thread = threading.Thread(target=cluster.workers[2].train, daemon=True)
        w4_thread = threading.Thread(target=cluster.workers[3].train, daemon=True)

        w1_thread.start()
        w2_thread.start()
        w3_thread.start()
        w4_thread.start()

        w1_thread.join()
        w2_thread.join()
        w3_thread.join()
        w4_thread.join()

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

        # if epoch % 10 == 0:
        #     tts('Eapock %d, ack %d' % (epoch, int(test_accuracy*100)))

        accuracies.append(test_accuracy)


        # Stop if accuracy has not risen 0.0001 above best acc in 400 epochs

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

    with open('eval_logs/custom_ps_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('%d workers, %d ps\n' % (config['num_workers'], config['num_ps']))
        outfile.write('784-128-10\n')
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        outfile.write('%d epochs before stop, %f accuracy delta, %d min epochs\n' % (epochs_before_stop, acc_delta, min_epochs))
        outfile.write('%d epochs, best accuracy: %f, epoch: %d\n\n' % (epoch, best_acc, best_acc_epoch))
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()


def main():
    load_config()

    cluster = Cluster(model_builder, dataset_fn, config)

    train(cluster)
    

main()