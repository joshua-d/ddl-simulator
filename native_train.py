import tensorflow as tf
import keras_model
from DatasetIterator import DatasetIterator
import time
import datetime


learning_rate = 0.1

num_epoches = 600
global_batch_size = 10

num_train_samples = 5000
num_test_samples = 5000


def dataset_fn():
    dataset = keras_model.mnist_dataset()
    dataset = dataset.shuffle(num_train_samples*10).take(num_train_samples)
    batch_size = global_batch_size

    return dataset, batch_size


steps_per_epoch = int(num_train_samples / global_batch_size)

model = keras_model.build_model()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


def forward_pass(batch):
    batch_inputs, batch_targets = batch
    with tf.GradientTape() as tape:
        predictions = model(batch_inputs, training=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )(batch_targets, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
            zip(grads, model.trainable_variables))


dataset, batch_size = dataset_fn()
dataset_iterator = DatasetIterator(dataset, batch_size)


def train():
    x_test, y_test = keras_model.test_dataset(num_test_samples)
    accuracies = []

    print('Beginning training')
    start_time = time.time()

    best_acc = 0
    acc_delta = 0.0001
    epochs_before_stop = 400
    epochs_under_delta = 0
    min_epochs = 200

    epoch = 0

    while True:
        epoch += 1

        for _ in range(steps_per_epoch):
            forward_pass(next(dataset_iterator))

        print('Finished epoch %d' % epoch)

        predictions = model.predict(x_test)

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

        # Stop if accuracy has not risen 0.001 above best acc in 30 epochs

        if epoch > min_epochs: 
            if test_accuracy > best_acc and test_accuracy - best_acc > acc_delta:
                best_acc = test_accuracy
                epochs_under_delta = 0
            else:
                epochs_under_delta += 1

            if epochs_under_delta >= epochs_before_stop:
                break


    time_elapsed = time.time() - start_time
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    best_acc = 0
    for acc in accuracies:
        if acc > best_acc:
            best_acc = acc

    with open('eval_logs/native_train_' + time_stamp + '.txt', 'w') as outfile:
        outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                        % (num_train_samples, num_test_samples, global_batch_size, learning_rate))
        outfile.write('%f seconds\n\n' % time_elapsed)
        outfile.write('%d epochs before stop, %f accuracy delta\n'% (epochs_before_stop, acc_delta))
        outfile.write('%d epochs, best accuracy: %f\n\n' % (epoch, best_acc))
        for accuracy in accuracies:
            outfile.write('%f\n' % accuracy)
        outfile.close()


train()