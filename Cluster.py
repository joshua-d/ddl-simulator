from ast import Param
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
from Node import UpdatePolicy


update_policy_str_map = {
    'replace': UpdatePolicy.REPLACE,
    'gradient': UpdatePolicy.GRADIENT,
    'average': UpdatePolicy.AVERAGE
}


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

        # TODO all node IDs must be independent - make sure worker and ps IDs don't collide
        # { Node id -> Node object }
        self.nodes = {}

        # Lists for easy access
        self.workers = []
        self.parameter_servers = []

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

        self._create_nodes()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_cond = threading.Condition()


    # TODO sync stuff, slow workers
    def _create_nodes(self):

        self.num_workers = 0
        self.num_slow_workers = 0
        self.num_ps = 0

        for node_desc in self.node_descs:

            _, params, _, build_optimizer = self.model_builder()

            # Build update_policies
            update_policies = {}
            for parent_id in node_desc['parents']:
                # TODO this line strictly assumes that node IDs are also their indexes in node_descs !!!!!
                update_policies[parent_id] = update_policy_str_map[self.node_descs[parent_id]['update_policy']]

            # Build param_locations
            # TODO this will be much different when model sharding is implemented
            param_locations = {}
            for parent_id in node_desc['parents']:
                param_locations[parent_id] = []
                for param_id in params:
                    param_locations[parent_id].append(param_id)

            # Add this node to children of parents
            # TODO assumes parents are already built
            # TODO currently unused, can remove
            for parent_id in node_desc['parents']:
                self.nodes[parent_id].children.append(node_desc['id'])


            if node_desc['node_type'] == 'ps':

                # Get update_interval
                if len(node_desc['parents']) > 0:
                    update_interval = node_desc['update_interval']
                else:
                    update_interval = 0

                ps = ParameterServer(
                    node_desc['id'], 
                    node_desc['parents'], 
                    update_policies, 
                    param_locations, 
                    self.ni, 
                    params, 
                    build_optimizer(self.learning_rate), 
                    update_interval
                )

                self.nodes[ps.id] = ps
                self.parameter_servers.append(ps)
                self.num_ps += 1

            elif node_desc['node_type'] == 'worker':

                # Build dataset_iterator
                dataset = self.dataset_fn(self.num_workers, self.num_train_samples) # TODO using num workers here is a bit hacky
                dataset_iterator = DatasetIterator(dataset, self.batch_size)

                if node_desc['slow']:
                    self.num_slow_workers += 1

                # TODO since we already call model builder above, could pass in individual things instead of model builder here
                worker = Worker(
                    node_desc['id'],
                    node_desc['parents'],
                    update_policies,
                    param_locations,
                    self.ni,
                    self.model_builder,
                    dataset_iterator,
                    self
                )

                self.nodes[worker.id] = worker
                self.workers.append(worker)
                self.num_workers += 1


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

        self.learning_rate = self._get_config_item(config, 'learning_rate')
        self.batch_size = self._get_config_item(config, 'batch_size')

        self.bypass_NI = self._get_config_item(config, 'bypass_NI')
        self.bandwidth = self._get_config_item(config, 'bandwidth')
        self.base_msg_time = self._get_config_item(config, 'base_msg_time')
        
        # Num train samples per epoch - passed into dataset_fn
        self.num_train_samples = self._get_config_item(config, 'num_train_samples')
        self.num_test_samples = self._get_config_item(config, 'num_test_samples')

        self.slow_worker_lb = self._get_config_item(config, 'slow_worker_lower_bound_ms')
        self.slow_worker_ub = self._get_config_item(config, 'slow_worker_upper_bound_ms')

        self.node_descs = self._get_config_item(config, 'nodes')

    def _get_config_item(self, config, item):
        if item not in config:
            raise Exception('%s not in config' % item)
        else:
            return config[item]


    def get_test_model(self):
        # TODO assumes node 0 is top level PS

        with self.nodes[0].params_lock:
            vals_by_param_id = self.nodes[0].get_params()

        for param_id in vals_by_param_id:
            self.test_model_params[param_id].assign(vals_by_param_id[param_id])

        return self.test_model


    def start(self):

        # Editable stopping condition vars
        max_epochs = 100
        acc_threshold = 0.955

        log_interval = 20 # log progress every 20 epochs


        # Init logging file
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        logging_filename = 'eval_logs/custom_ps_%s.txt' % time_stamp

        with open(logging_filename, 'w') as outfile:
            outfile.write('%d workers, %d ps\n' % (self.num_workers, self.num_ps))
            outfile.write('%d slow workers, %d to %d ms\n' % (self.num_slow_workers, self.slow_worker_lb, self.slow_worker_ub))

            # MODEL INFO
            outfile.write('784-128-10\n')

            if self.bypass_NI:
                outfile.write('NETWORK INTERFACE BYPASSED\n')
            else:
                outfile.write('%d bandwidth, %f base msg time\n' % (self.bandwidth, self.base_msg_time))

            outfile.write('num train samples: %d, num test samples: %d, batch size: %d, learning rate: %f\n'
                            % (self.num_train_samples, self.num_test_samples, self.batch_size, self.learning_rate))
            outfile.write('%f acc threshold, %d max epochs\n\n' % (acc_threshold, max_epochs))
            outfile.close()
        

        # Logging and eval vars
        x_test, y_test = keras_model.test_dataset(self.num_test_samples)
        accuracies = []

        best_acc = 0
        best_acc_epoch = 0
        epoch = 0
        steps_per_epoch = int(self.num_train_samples / self.batch_size)

        
        # Start nodes
        for node in self.nodes.values():
            node.start()


        # Begin training
        print('Beginning training')
        start_time = time.time()

        # Start network emulator
        self.ni.start()

        while True:
            epoch += 1

            # Schedule steps for this epoch
            self.steps_scheduled = steps_per_epoch
            
            # Wait for workers to complete scheduled steps
            with self.steps_completed_cond:
                while self.steps_completed < self.steps_scheduled:
                    self.steps_completed_cond.wait()
                # print("Steps completed: %d" % self.steps_completed)
                self.steps_completed = 0
                self.steps_scheduled = 0


            print('Finished epoch %d' % epoch)

            
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
            if epoch % log_interval == 0:
                with open(logging_filename, 'a') as outfile:
                    for accuracy in accuracies:
                        outfile.write('%f\n' % accuracy)
                    outfile.close()
                accuracies = []


            # STOPPING CONDITIONS 
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_acc_epoch = epoch

            if test_accuracy >= acc_threshold or epoch >= max_epochs:
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
            outfile.write('%d epochs, best accuracy: %f, epoch: %d\n%f seconds\n\n' % (epoch, best_acc, best_acc_epoch, time_elapsed))
            
            for worker in self.workers:
                outfile.write('Worker %d: %d steps\n' % (worker.id, worker.steps_completed))
            
            outfile.write('\n')
            outfile.write(data)
            outfile.close()