import tensorflow as tf
import numpy as np
import threading

# Only imported for test dataset
import keras_model
import time
import datetime

from ParameterServer import ParameterServer
from SyncParameterServer import SyncParameterServer
from Worker import Worker
from DatasetIterator import DatasetIterator
from NetworkInterface import NetworkInterface, NetworkInterfaceBypass


# For now assuming cluster is the outermost name for the system, i. e. only one cluster per simulator run
class Cluster:

    def __init__(self, model_builder, dataset_fn, config):

        # f() -> (model, params, forward_pass, build_optimizer)
        #  model = keras model
        #  params = { param_id -> model var }
        #  forward_pass = f(batch) -> { param_id: param gradient }
        #  build_optimizer = f(learning_rate) -> optimizer
        #    optimizer = object with apply_gradients([(grad, model var)]), usually a tf.keras.optimizers.X
        self.model_builder = model_builder

        # TODO may want to have dataset_fn return an infinite iterator over worker's dataset
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

        self.test_model, self.test_model_params, _, _ = model_builder()

        if self.bypass_NI:
            self.ni = NetworkInterfaceBypass(self)
        else:
            msg_size = self._get_model_size()
            self._set_bandwidth(msg_size)
            self.ni = NetworkInterface(self, self.bandwidth, msg_size, msg_size)

        self._create_parameter_servers()
        self._create_workers()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_cond = threading.Condition()



    def _create_parameter_servers(self):

        # Get a copy of the model and place its params on the parameter server(s)
        _, params, _, build_optimizer = self.model_builder()

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
                self.parameter_servers[ps_id] = ParameterServer(ps_id, params_objs[i], build_optimizer(self.learning_rate), self.ni)
            elif self.training_style == 'sync':
                self.parameter_servers[ps_id] = SyncParameterServer(ps_id, params_objs[i], build_optimizer(self.learning_rate), self.ni, self.num_workers)

            self.param_locations[ps_id] = list(params_objs[i].keys())

    
    def _create_workers(self):
        for i in range(self.num_workers):
            dataset = self.dataset_fn(i, self.num_train_samples)
            dataset_iterator = DatasetIterator(dataset, self.batch_size)
            
            self.workers.append(Worker(i, self.model_builder, dataset_iterator, self.param_locations, self.ni, self))


    # Returns the size of the model in bits
    # Used for setting the params and grads msg size for NI
    def _get_model_size(self):
        total_bits = 0
        for param in self.test_model_params.values():
            total_bits += np.size(param) * np.size(param.dtype) * 8
        return total_bits


    # If a base_msg_time was given in config, sets bandwidth accordingly
    # Assumes all msgs are the same size
    def _set_bandwidth(self, msg_size):
        if self.base_msg_time != 0:
            self.bandwidth = msg_size / self.base_msg_time


    def _parse_config(self, config):
        self.num_ps = self._get_config_item(config, 'num_ps')
        self.num_workers = self._get_config_item(config, 'num_workers')
        self.training_style = self._get_config_item(config, 'training_style')
        
        self.learning_rate = self._get_config_item(config, 'learning_rate')
        self.batch_size = self._get_config_item(config, 'batch_size')

        self.bypass_NI = self._get_config_item(config, 'bypass_NI')
        self.bandwidth = self._get_config_item(config, 'bandwidth')
        self.base_msg_time = self._get_config_item(config, 'base_msg_time')
        
        # Num train samples per epoch - passed into dataset_fn
        self.num_train_samples = self._get_config_item(config, 'num_train_samples')
        self.num_test_samples = self._get_config_item(config, 'num_test_samples')

        self.num_slow_workers = self._get_config_item(config, 'num_slow_workers')
        self.slow_worker_lb = self._get_config_item(config, 'slow_worker_lower_bound_ms') / 1000
        self.slow_worker_ub = self._get_config_item(config, 'slow_worker_upper_bound_ms') / 1000

        if self.training_style == 'sync': # TODO document or remove this
            self.num_slow_workers = 0

        if self.training_style == 'sync' and self.num_ps > 1:
            raise Exception('More than 1 PS with synchronous training is not supported')

    def _get_config_item(self, config, item):
        if item not in config:
            raise Exception('%s not in config' % item)
        else:
            return config[item]


    def get_test_model(self):
        params_msgs = []
        for ps in self.parameter_servers.values():
            with ps.params_lock:
                params_msgs.append(ps.get_params())

        for vals_by_param_id in params_msgs:
            for param_id in vals_by_param_id:
                self.test_model_params[param_id].assign(vals_by_param_id[param_id])

        return self.test_model


    def start(self):

        # Editable stopping condition vars
        max_epochs = 12
        acc_threshold = 0.95
        eval_interval = 100 # eval every 100 batches
        log_interval = 50 # log progress every 20 eval_intervals

        batches_per_epoch = int(self.num_train_samples / self.batch_size)
        max_eval_intervals = int((batches_per_epoch / eval_interval) * max_epochs)
        max_batches = max_eval_intervals * eval_interval

        reached_92 = False
        reached_92_batches = 0

        # Init logging file
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        logging_filename = 'eval_logs/sim_%s_%s.txt' % (self.training_style, time_stamp)

        with open(logging_filename, 'w') as outfile:
            outfile.write('%d workers, %d ps\n' % (self.num_workers, self.num_ps))
            outfile.write('%d slow workers, %d to %d ms\n' % (self.num_slow_workers, self.slow_worker_lb*1000, self.slow_worker_ub*1000))
            outfile.write('%s training\n' % self.training_style)

            # MODEL INFO
            outfile.write('784-256-256-256-256-256-256-10\n')

            if self.bypass_NI:
                outfile.write('NETWORK INTERFACE BYPASSED\n')
            else:
                outfile.write('%d bandwidth, %f base msg time\n' % (self.bandwidth, self.base_msg_time))

            outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                            % (self.num_train_samples, self.num_test_samples, self.batch_size, self.learning_rate))
            outfile.write('%f acc threshold, %d max epochs (%d max batches)\n' % (acc_threshold, max_epochs, max_batches))
            outfile.write('eval interval: %d batches\n\n' % eval_interval)
            outfile.close()
        

        # Eval vars
        x_test, y_test = keras_model.test_dataset(self.num_test_samples)
        accuracies = []

        
        # Create and start worker threads
        worker_threads = []
        
        for worker in self.workers:
            worker_thread = threading.Thread(target=worker.start, daemon=True)
            worker_threads.append(worker_thread)

        for wt in worker_threads:
            wt.start()

        # Create and start PS threads
        ps_threads = []

        for ps_id in self.parameter_servers:
                ps = self.parameter_servers[ps_id]
                ps_thread = threading.Thread(target=ps.start, daemon=True)
                ps_threads.append(ps_thread)

        for pst in ps_threads:
            pst.start()


        # Begin training
        print('Beginning training')
        start_time = time.time()

        # Start network emulator
        self.ni.start()

        eval_num = 0

        while True:
            eval_num += 1

            # Schedule steps for this epoch
            self.steps_scheduled = eval_interval
            
            # Wait for workers to complete scheduled steps
            with self.steps_completed_cond:
                while self.steps_completed < self.steps_scheduled:
                    self.steps_completed_cond.wait()
                # print("Steps completed: %d" % self.steps_completed)
                self.steps_completed = 0
                self.steps_scheduled = 0


            print('Finished eval %d (%d batches)' % (eval_num, eval_num*eval_interval))

            
            # Evaluate model
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

            test_accuracy = float(num_correct) / self.num_test_samples
            print('Test accuracy: %f' % test_accuracy)

            accuracies.append(test_accuracy)


            # Log
            if eval_num % log_interval == 0:
                with open(logging_filename, 'a') as outfile:
                    for accuracy in accuracies:
                        outfile.write('%f\n' % accuracy)
                    outfile.close()
                accuracies = []


            # STOPPING CONDITIONS
            if not reached_92 and test_accuracy >= 0.92:
                reached_92 = True
                reached_92_batches = eval_num * eval_interval

            if test_accuracy >= acc_threshold or eval_num >= max_eval_intervals:
                break


        # Training done, complete logging
        time_elapsed = time.time() - start_time
        
        with open(logging_filename, 'a') as outfile:
            for accuracy in accuracies:
                outfile.write('%f\n' % accuracy)
            outfile.close()

        with open(logging_filename, 'r+') as outfile:
            data = outfile.read()
            outfile.seek(0)
            outfile.write('95: %d batches, 92: %d batches, %f epochs\n%f seconds\n\n' % (eval_num*eval_interval, reached_92_batches, eval_num*eval_interval / batches_per_epoch, time_elapsed))
            
            for worker in self.workers:
                outfile.write('Worker %d: %d steps\n' % (worker.id, worker.steps_completed))
            
            outfile.write('\n')
            outfile.write(data)
            outfile.close()