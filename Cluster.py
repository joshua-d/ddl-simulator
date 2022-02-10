import tensorflow as tf
import threading

# Only imported for test dataset
import keras_model
import time
import datetime

from ParameterServer import ParameterServer
from SyncParameterServer import SyncParameterServer
from Worker import Worker
from SyncWorker import SyncWorker
from DatasetIterator import DatasetIterator


# TODO don't just pass config obj into Cluster, have a fn parse it into class fields

# For now assuming cluster is the outermost name for the system, i. e. only one cluster per simulator run
class Cluster:

    def __init__(self, model_builder, dataset_fn, config):

        # f() -> (model, params, forward_pass)
        #  model = keras model
        #  params = { param_id -> model var }
        #  forward_pass = f(batch) -> gradients
        self.model_builder = model_builder

        # f(worker_id) -> worker's dataset
        self.dataset_fn = dataset_fn

        # TODO load each config item into class field?
        self.config = config

        # TODO dict for consistency with self.parameter_servers
        # [ Worker ]
        self.workers = []

        # [ Thread ]
        self.worker_threads = []

        # { PS_id -> ParameterServer }
        self.parameter_servers = {}

        # { PS_id -> [v1_id, v2_id, ...] }
        self.param_locations = {}

        self._create_parameter_servers()
        self._create_workers()

        self.test_model, self.test_model_params, _ = model_builder()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_lock = threading.Lock()


    def _create_parameter_servers(self):
        num_ps = self._check_config_item('num_ps')
        learning_rate = self._check_config_item('learning_rate')
        training_style = self._check_config_item('training_style')

        _, params, _ = self.model_builder()

        # round robin placement
        params_objs = []
        for i in range(num_ps):
            params_objs.append({})

        next_ps_ind = 0
        for param_id in params:
            params_objs[next_ps_ind][param_id] = params[param_id]
            next_ps_ind = (next_ps_ind + 1) % num_ps

        for i in range(num_ps):
            ps_id = 'ps%d' % i
            if training_style == 'async':
                self.parameter_servers[ps_id] = ParameterServer(params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=learning_rate))
            elif training_style == 'sync':
                self.parameter_servers[ps_id] = SyncParameterServer(params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=learning_rate), self.workers, self)

            self.param_locations[ps_id] = list(params_objs[i].keys())

    
    def _create_workers(self):
        num_workers = self._check_config_item('num_workers')
        training_style = self._check_config_item('training_style')

        for i in range(num_workers):
            dataset, batch_size = self.dataset_fn(i)
            dataset_iterator = DatasetIterator(dataset, batch_size)
            if training_style == 'async':
                self.workers.append(Worker(self, i, self.model_builder, dataset_iterator))
            elif training_style == 'sync':
                self.workers.append(SyncWorker(self, i, self.model_builder, dataset_iterator))

    # TODO maybe rename to get
    def _check_config_item(self, item):
        if self.config[item] is None:
            raise Exception('%s not in config' % item)
        else:
            return self.config[item]


    def get_test_model(self):
        for ps_id in self.param_locations:
            ps = self.parameter_servers[ps_id]
            ps.params_lock.acquire()
            params = ps.on_request()
            for param_id in params:
                self.test_model_params[param_id].assign(params[param_id])
            ps.params_lock.release()

        return self.test_model


    # TODO implement configurable stopping/training conditions/callbacks
    def train(self):
        num_train_samples = self._check_config_item('num_train_samples')
        num_test_samples = self._check_config_item('num_test_samples')
        batch_size = self._check_config_item('batch_size')

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
        steps_per_epoch = int(num_train_samples / batch_size)

        while True:
            epoch += 1

            self.steps_completed = 0
            self.steps_scheduled = steps_per_epoch

            self.worker_threads = []
            for worker in self.workers:
                worker_thread = threading.Thread(target=worker.train, daemon=True)
                self.worker_threads.append(worker_thread)

            for wt in self.worker_threads:
                wt.start()

            for wt in self.worker_threads:
                wt.join()

            print('Finished epoch %d' % epoch)

            predictions = self.get_test_model().predict(x_test)

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

            # stop conditions 
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

        with open('eval_logs/custom_ps_%s_%s.txt' % (self.config['training_style'], time_stamp), 'w') as outfile:
            outfile.write('%d workers, %d ps\n' % (self.config['num_workers'], self.config['num_ps']))
            outfile.write('%s training\n' % self.config['training_style'])
            outfile.write('784-128-10\n')
            outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                            % (num_train_samples, num_test_samples, batch_size, self.config['learning_rate']))
            outfile.write('%f seconds\n\n' % time_elapsed)
            outfile.write('%d epochs before stop, %f accuracy delta, %d min epochs\n' % (epochs_before_stop, acc_delta, min_epochs))
            outfile.write('%d epochs, best accuracy: %f, epoch: %d\n\n' % (epoch, best_acc, best_acc_epoch))
            for accuracy in accuracies:
                outfile.write('%f\n' % accuracy)
            outfile.close()