from ast import Param
import tensorflow as tf
import numpy as np
import threading
from math import inf

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
        # f(num_workers, worker_idx, num_train_samples) -> worker's dataset
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
            self.ni = NetworkInterface(self, self._create_node_bws(), msg_size, msg_size)

        self._create_nodes()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_cond = threading.Condition()


    def _create_node_bws(self):
        inbound_bws = {}
        outbound_bws = {}

        for node_desc in self.node_descs:
            inbound_bws[node_desc['id']] = node_desc['inbound_bw'] * 1000000
            outbound_bws[node_desc['id']] = node_desc['outbound_bw'] * 1000000

        return (inbound_bws, outbound_bws)


    # TODO slow workers
    def _create_nodes(self):

        self.num_slow_workers = 0
        self.num_ps = 0
        self.num_workers = 0

        # Build dataset_iterator
        dataset = self.dataset_fn(self.num_train_samples)
        dataset_iterator = DatasetIterator(dataset, self.batch_size, self.data_chunk_size)

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

            # Add this node to children of parents, used for sync tracking
            # TODO assumes parents are already built
            for parent_id in node_desc['parents']:
                self.nodes[parent_id].children.append(node_desc['id'])


            if node_desc['node_type'] == 'ps':

                # Get update_interval
                if len(node_desc['parents']) > 0:
                    update_interval = node_desc['update_interval']
                else:
                    update_interval = inf # TODO make sure top level PS never updates

                # Set async or sync
                if node_desc['train_style'] == 'async':
                    PSClass = ParameterServer
                elif node_desc['train_style'] == 'sync':
                    PSClass = SyncParameterServer

                # Get this node's update policy
                update_policy = update_policy_str_map[node_desc['update_policy']]

                ps = PSClass(
                    node_desc['id'], 
                    node_desc['parents'], 
                    update_policy,
                    update_policies, 
                    param_locations, 
                    self.ni, 
                    params, 
                    build_optimizer(self.learning_rate), 
                    update_interval,
                    self.ps_return_threshold
                )

                self.nodes[ps.id] = ps
                self.parameter_servers.append(ps)
                self.num_ps += 1

            elif node_desc['node_type'] == 'worker':

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
                    build_optimizer(self.learning_rate),
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


    def _parse_config(self, config):

        self.learning_rate = self._get_config_item(config, 'learning_rate')
        self.batch_size = self._get_config_item(config, 'batch_size')

        self.bypass_NI = self._get_config_item(config, 'bypass_NI')
        
        # Num train samples per epoch - passed into dataset_fn
        self.num_train_samples = self._get_config_item(config, 'num_train_samples')
        self.num_test_samples = self._get_config_item(config, 'num_test_samples')

        self.slow_worker_lb = self._get_config_item(config, 'slow_worker_lower_bound_ms')
        self.slow_worker_ub = self._get_config_item(config, 'slow_worker_upper_bound_ms')

        self.node_descs = self._get_config_item(config, 'nodes')

        self.ps_return_threshold = self._get_config_item(config, 'ps_return_threshold')

        self.data_chunk_size = self._get_config_item(config, 'data_chunk_size')

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
        max_epochs = 12
        acc_threshold = 0.95
        eval_interval = 100 # eval every 100 batches
        log_interval = 50 # log progress every 20 eval_intervals

        batches_per_epoch = int(self.num_train_samples / self.batch_size)
        max_eval_intervals = int((batches_per_epoch / eval_interval) * max_epochs)
        max_batches = max_eval_intervals * eval_interval


        # Init logging file
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        logging_filename = 'eval_logs/sim_%s.txt' % (time_stamp)

        _, _, _, build_optimizer = self.model_builder()
        optimizer = build_optimizer(self.learning_rate)

        with open(logging_filename, 'w') as outfile:
            outfile.write('%d workers, %d ps\n' % (self.num_workers, self.num_ps))
            outfile.write('%d slow workers, %d to %d ms\n' % (self.num_slow_workers, self.slow_worker_lb*1000, self.slow_worker_ub*1000))

            # MODEL INFO
            outfile.write('784-256-256-256-256-256-256-10\n')

            # Optimizer type
            outfile.write('optimizer: ' + str(type(optimizer)) + '\n')

            if self.bypass_NI:
                outfile.write('NETWORK INTERFACE BYPASSED\n')

            outfile.write('num train samples: %d, num test samples: %d\nbatch size: %d, learning rate: %f\n'
                            % (self.num_train_samples, self.num_test_samples, self.batch_size, self.learning_rate))
            outfile.write('%f acc threshold, %d max epochs (%d max batches)\n' % (acc_threshold, max_epochs, max_batches))
            outfile.write('eval interval: %d batches\n' % eval_interval)
            outfile.write('ps return threshold: %f\n\n' % self.ps_return_threshold)
            outfile.close()
        

        # Eval vars
        x_test, y_test = keras_model.test_dataset(self.num_test_samples)
        accuracies = []

        
        # Start nodes
        for node in self.nodes.values():
            node.start()


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
            outfile.write('%d batches, %f epochs\n%f seconds\n\n' % (eval_num*eval_interval, eval_num*eval_interval / batches_per_epoch, time_elapsed))
            
            for worker in self.workers:
                outfile.write('Worker %d: %d steps\n' % (worker.id, worker.steps_completed))
            
            outfile.write('\n')
            outfile.write(data)
            outfile.write('\n[\n')

            for node_desc in self.node_descs:
                outfile.write('\t' + str(node_desc) + '\n')

            outfile.write(']\n')
            outfile.close()