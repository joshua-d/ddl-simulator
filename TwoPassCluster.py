import datetime
import numpy as np
from DatasetIterator import DatasetIterator
from NetworkSequenceGenerator import NetworkSequenceGenerator, WorkerStepEvent, SendParamsEvent, ReceiveParamsEvent, PSAggrEvent, PSApplyEvent, PSParentApplyEvent
import keras_model


class Worker:
    def __init__(self, id, params, forward_pass, dataset_iterator, optimizer):
        self.id = id

        self.params = params
        self.forward_pass = forward_pass
        self.dataset_iterator = dataset_iterator
        self.optimizer = optimizer

        # List of param sets
        self.incoming_parent_params = []

        # Dataset stuff
        chunk = next(dataset_iterator)
        self.data_chunk_size = len(chunk)
        self.data_chunk_iterator = iter(chunk)
        self.batch_idx = 0

        self.steps_complete = 0

    def get_next_batch(self):
        batch = next(self.data_chunk_iterator)
        self.batch_idx += 1

        if self.batch_idx == self.data_chunk_size:
            chunk = next(self.dataset_iterator)
            self.data_chunk_size = len(chunk)
            self.data_chunk_iterator = iter(chunk)
            self.batch_idx = 0

        return batch

    def train_step(self):
        gradients = self.forward_pass(self.get_next_batch())
        self.optimizer.apply_gradients(zip(gradients, self.params.values()))
        self.steps_complete += 1

    def replace_params(self, param_set):
        for param_id in param_set:
            self.params[param_id].assign(param_set[param_id])

    def get_params(self):
        params = {}
        for param_id in self.params:
            params[param_id] = self.params[param_id].value()
        return params


class ParameterServer:
    def __init__(self, id, parent, sync_style, params):
        self.id = id
        self.parent = parent
        self.sync_style = sync_style
        self.params = params

        # Lists of param sets
        self.incoming_child_params = []
        self.incoming_parent_params = []

        self.received_first_update = False
        self.has_async_child = False

        self.tsync_total = 0
        self.tsync_n = 0
        self.tsync_last = 0

    # Sync
    def aggr_and_apply_params(self, param_sets):
        # Average and assign params
        for param_id in self.params:
            param_value = 0

            for param_set in param_sets:
                param_value += param_set[param_id]
            
            param_value /= len(param_sets)

            if not self.has_async_child:
                self.params[param_id].assign(param_value)
            else:
                # Average into current instead of replacing
                self.params[param_id].assign((param_value + self.params[param_id].value()) / 2)

    # Async
    def apply_params(self, param_set):
        if self.parent is not None or not self.received_first_update:
            # Assign params for relay (up or down!)
            for param_id in self.params:
                self.params[param_id].assign(param_set[param_id])

            self.received_first_update = True
        else:
            # Average params into current (async only works on one set of params at a time)
            # Only top level ps does this
            for param_id in self.params:
                param_value = (self.params[param_id].value() + param_set[param_id]) / 2
                self.params[param_id].assign(param_value)

    def get_params(self):
        params = {}
        for param_id in self.params:
            params[param_id] = self.params[param_id].value()
        return params
            
            
class TwoPassCluster:

    def __init__(self, model_builder, dataset_fn, config):
        self.model_builder = model_builder
        self.dataset_fn = dataset_fn
        self.config = config

        self.nodes = {}

        self._parse_config(config)

        self.test_model, self.test_model_params, _, _ = model_builder()

        self._create_nodes()

        self.steps_complete = 0

        msg_size = self._get_model_size()
        self.nsg = NetworkSequenceGenerator(self.node_descs, msg_size, self.network_style == 'hd')
        self.gen_buf = 100
        self.n_generated = 0

    def _create_nodes(self):

        # Build dataset_iterator
        dataset = self.dataset_fn(self.num_train_samples)
        dataset_iterator = DatasetIterator(dataset, self.batch_size, self.data_chunk_size)

        for node_desc in self.node_descs:

            _, params, forward_pass, build_optimizer = self.model_builder()

            if node_desc['node_type'] == 'ps':
                ps = ParameterServer(node_desc['id'], node_desc['parent'], node_desc['sync_style'], params)
                self.nodes[ps.id] = ps
                if node_desc['sync_style'] == 'async' and node_desc['parent'] is not None:
                    self.nodes[node_desc['parent']].has_async_child = True

            elif node_desc['node_type'] == 'worker':
                worker = Worker(node_desc['id'], params, forward_pass, dataset_iterator, build_optimizer(self.learning_rate))
                self.nodes[worker.id] = worker

    def _parse_config(self, config):

        self.learning_rate = self._get_config_item(config, 'learning_rate')
        self.batch_size = self._get_config_item(config, 'batch_size')

        self.bypass_NI = self._get_config_item(config, 'bypass_NI')
        
        # Num train samples per epoch - passed into dataset_fn
        self.num_train_samples = self._get_config_item(config, 'num_train_samples')
        self.num_test_samples = self._get_config_item(config, 'num_test_samples')

        self.network_style = self._get_config_item(config, 'network_style')

        self.node_descs = self._get_config_item(config, 'nodes')

        self.ps_return_threshold = 0 # TODO removed functionality

        self.data_chunk_size = self._get_config_item(config, 'data_chunk_size')

        self.acc_thresholds = self._get_config_item(config, 'acc_thresholds')
        self.eval_interval = self._get_config_item(config, 'eval_interval')
        self.max_epochs = self._get_config_item(config, 'max_epochs')

        self.generate_gantt = self._get_config_item(config, 'generate_gantt')

    def _get_config_item(self, config, item):
        if item not in config:
            raise Exception('%s not in config' % item)
        else:
            return config[item]

    def get_test_model(self):
        # assumes node 0 is top level PS
        vals_by_param_id = self.nodes[0].get_params()

        for param_id in vals_by_param_id:
            self.test_model_params[param_id].assign(vals_by_param_id[param_id])

        return self.test_model

    def _get_model_size(self):
        total_bits = 0
        for param in self.test_model_params.values():
            total_bits += np.size(param) * np.size(param.dtype) * 8
        return total_bits

    def process_event(self, event):

        if type(event) == SendParamsEvent:
            params = self.nodes[event.sender_id].get_params()
            node = self.nodes[event.receiver_id]
            if type(node) == Worker or node.parent == event.sender_id:
                node.incoming_parent_params.append(params)
            else:
                node.incoming_child_params.append(params)

        elif type(event) == ReceiveParamsEvent:
            receiver = self.nodes[event.receiver_id]
            if type(receiver) == Worker:
                # Worker should only have 1 param set in incoming params
                params = receiver.incoming_parent_params[0]
                receiver.incoming_parent_params = []
                receiver.replace_params(params)

        elif type(event) == WorkerStepEvent:
            self.nodes[event.worker_id].train_step()
            self.steps_complete += 1

        elif type(event) == PSApplyEvent:
            ps = self.nodes[event.ps_id]
            if ps.sync_style == 'async':
                params = ps.incoming_child_params.pop(0)
                ps.apply_params(params)
            elif ps.sync_style == 'sync':
                param_sets = ps.incoming_child_params
                ps.incoming_child_params = []
                ps.aggr_and_apply_params(param_sets)

                # Measure tsync
                ps.tsync_total += event.start_time - ps.tsync_last
                ps.tsync_last = event.start_time
                ps.tsync_n += 1

        elif type(event) == PSParentApplyEvent:
            ps = self.nodes[event.ps_id]
            params = ps.incoming_parent_params.pop(0)
            ps.apply_params(params)

    def start(self):

        # Prepare vars
        log_interval = 50

        batches_per_epoch = int(self.num_train_samples / self.batch_size)
        max_eval_intervals = int((batches_per_epoch / self.eval_interval) * self.max_epochs)

        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        logging_filename = 'eval_logs/sim_%s.txt' % (time_stamp)

        if self.generate_gantt:
            saved_events = []

        # Eval vars
        x_test, y_test = keras_model.test_dataset(self.num_test_samples)
        accuracies = []
        threshold_results = []

        # Begin training
        print('Beginning training')
        next_steps_milestone = self.eval_interval
        eval_num = 0

        while True:
            eval_num += 1

            # Process events until next steps milestone
            while self.steps_complete < next_steps_milestone:
                if len(self.nsg.events) == 0:
                    for _ in range(self.gen_buf):
                        self.nsg.generate()
                    self.n_generated += self.gen_buf
                current_event = self.nsg.events.pop(0)
                if self.generate_gantt:
                    saved_events.append(current_event)
                self.process_event(current_event)

            next_steps_milestone += self.eval_interval

            print('Finished %d steps' % self.steps_complete)

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

            # Check if next acc threshold has been reached
            if test_accuracy >= self.acc_thresholds[0]:
                epochs = self.steps_complete / batches_per_epoch
                # Mark end time of last step event of this interval
                threshold_results.append((self.acc_thresholds.pop(0), epochs, self.steps_complete, current_event.end_time))

            # STOPPING CONDITIONS
            if len(self.acc_thresholds) == 0 or eval_num >= max_eval_intervals:
                break


        # Training done, complete logging
        
        with open(logging_filename, 'a') as outfile:
            for accuracy in accuracies:
                outfile.write('%f\n' % accuracy)
            outfile.close()

        with open(logging_filename, 'r+') as outfile:
            data = outfile.read()
            outfile.seek(0)

            for res in threshold_results:
                outfile.write('%f:\n%f epochs\n%d batches\n%f end time\n\n' % res)
            
            for node in self.nodes.values():
                if type(node) == Worker:
                    outfile.write('Worker %d: %d steps\n' % (node.id, node.steps_complete))
            
            outfile.write('\n')

            for node in self.nodes.values():
                if type(node) == ParameterServer and node.sync_style == 'sync':
                    outfile.write('PS %d tsync: %f\n' % (node.id, node.tsync_total / node.tsync_n))

            outfile.write('\n')
            outfile.write(data)

            outfile.write('\n')
            for k in self.config:
                if k == 'nodes' or k == 'acc_thresholds':
                    continue
                outfile.write('%s: %s\n' % (k, self.config[k]))

            outfile.write('\n[\n')

            for node_desc in self.node_descs:
                outfile.write('\t' + str(node_desc) + '\n')

            outfile.write(']\n')
            outfile.close()

        if self.generate_gantt:
            self.nsg.events = saved_events
            self.nsg.generate_gantt(time_stamp)
