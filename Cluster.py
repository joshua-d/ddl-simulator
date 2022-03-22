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
from NodeCommunication import NodeCommunication



# For now assuming cluster is the outermost name for the system, i. e. only one cluster per simulator run
class Cluster:

    def __init__(self, model_builder, dataset_fn, config):

        # f() -> (model, params, forward_pass)
        #  model = keras model
        #  params = { param_id -> model var }
        #  forward_pass = f(batch) -> { param_id: param gradient }
        self.model_builder = model_builder

        # f(worker_id, num_train_samples) -> worker's dataset
        self.dataset_fn = dataset_fn

        # TODO dict for consistency with self.parameter_servers
        # [ Worker ]
        self.workers = []

        # { PS_id -> ParameterServer }
        self.parameter_servers = {}

        # { PS_id -> [v1_id, v2_id, ...] }
        self.param_locations = {}


        self._parse_config(config)

        self.nc = NodeCommunication(self)

        self._create_parameter_servers()
        self._create_workers()

        self.test_model, self.test_model_params, _ = model_builder()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_lock = threading.Lock()

        self.print_lock = threading.Lock()


    def _create_parameter_servers(self):

        # Get a copy of the model and place its params on the parameter server(s)
        _, params, _ = self.model_builder()

        # round robin placement
        params_objs = []
        for i in range(self.num_ps):
            params_objs.append({})

        next_ps_ind = 0
        for param_id in params:
            params_objs[next_ps_ind][param_id] = params[param_id]
            next_ps_ind = (next_ps_ind + 1) % self.num_ps

        for i in range(self.num_ps):
            ps_id = 'ps%d' % i
            if self.training_style == 'async':
                self.parameter_servers[ps_id] = ParameterServer(ps_id, params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), self.nc)
            elif self.training_style == 'sync':
                self.parameter_servers[ps_id] = SyncParameterServer(ps_id, params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), self.nc, self.num_workers)

            self.param_locations[ps_id] = list(params_objs[i].keys())

    
    def _create_workers(self):
        for i in range(self.num_workers):
            dataset = self.dataset_fn(i, self.num_train_samples)
            dataset_iterator = DatasetIterator(dataset, self.batch_size)
            
            if self.training_style == 'async':
                self.workers.append(Worker(i, self.model_builder, dataset_iterator, self.param_locations, self.nc, self))
            elif self.training_style == 'sync':
                self.workers.append(SyncWorker(i, self.model_builder, dataset_iterator, self.param_locations, self.nc, self))


    def _parse_config(self, config):
        self.num_ps = self._get_config_item(config, 'num_ps')
        self.num_workers = self._get_config_item(config, 'num_workers')
        self.training_style = self._get_config_item(config, 'training_style')
        
        self.learning_rate = self._get_config_item(config, 'learning_rate')
        self.batch_size = self._get_config_item(config, 'batch_size')
        
        # Num train samples per epoch - passed into dataset_fn
        self.num_train_samples = self._get_config_item(config, 'num_train_samples')

        self.num_test_samples = self._get_config_item(config, 'num_test_samples')

        if self.training_style == 'sync' and self.num_ps > 1:
            raise Exception('More than 1 PS with synchronous training is not supported')

    def _get_config_item(self, config, item):
        if item not in config:
            raise Exception('%s not in config' % item)
        else:
            return config[item]


    def get_test_model(self):
        # TODO this is pretty hacky - forces worker 0 to request params, getting latest
        params_msgs = self.nc.request_params_and_wait(self.workers[0])
        for vals_by_param_id in params_msgs:
            for param_id in vals_by_param_id:
                self.test_model_params[param_id].assign(vals_by_param_id[param_id])

        return self.test_model


    def start(self):
        x_test, y_test = keras_model.test_dataset(self.num_test_samples)
        accuracies = []

        # Editable stopping condition vars
        max_epochs = 400
        acc_threshold = 0.95

        best_acc = 0
        best_acc_epoch = 0

        epoch = 0
        steps_per_epoch = int(self.num_train_samples / self.batch_size)


        ps_threads = []

        for ps_id in self.parameter_servers:
                ps = self.parameter_servers[ps_id]
                ps_thread = threading.Thread(target=ps.start, daemon=True)
                ps_threads.append(ps_thread)

        for pst in ps_threads:
            pst.start()


        print('Beginning training')
        start_time = time.time()

        while True:
            epoch += 1

            self.steps_completed = 0
            self.steps_scheduled = steps_per_epoch

            worker_threads = []
            
            for worker in self.workers:
                worker_thread = threading.Thread(target=worker.start, daemon=True)
                worker_threads.append(worker_thread)

            for wt in worker_threads:
                wt.start()

            for wt in worker_threads:
                wt.join()

            print('Finished epoch %d' % epoch)
            
            predictions = self.get_test_model().predict(x_test)

            if self.training_style == 'sync':
                self.parameter_servers['ps0'].reset_round()

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

            test_accuracy = float(num_correct) / self.num_test_samples
            print('Test accuracy: %f' % test_accuracy)

            accuracies.append(test_accuracy)


            # STOPPING CONDITIONS 
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_acc_epoch = epoch

            if test_accuracy > acc_threshold or epoch >= max_epochs:
                break


        time_elapsed = time.time() - start_time
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        with open('eval_logs/custom_ps_%s_%s.txt' % (self.training_style, time_stamp), 'w') as outfile:
            outfile.write('%d workers, %d ps\n' % (self.num_workers, self.num_ps))
            outfile.write('%s training\n' % self.training_style)
            outfile.write('784-128-10\n')
            outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                            % (self.num_train_samples, self.num_test_samples, self.batch_size, self.learning_rate))
            outfile.write('%f seconds\n\n' % time_elapsed)
            outfile.write('%f acc threshold, %d max epochs\n' % (acc_threshold, max_epochs))
            outfile.write('%d epochs, best accuracy: %f, epoch: %d\n\n' % (epoch, best_acc, best_acc_epoch))
            for accuracy in accuracies:
                outfile.write('%f\n' % accuracy)
            outfile.close()