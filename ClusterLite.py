from threading import Thread, Condition
from DatasetIterator import DatasetIterator
from NetworkSequenceGenerator import *
import keras_model


class Worker:
    def __init__(self, id, params, forward_pass, dataset_iterator, optimizer):
        self.id = id

        self.params = params
        self.forward_pass = forward_pass
        self.dataset_iterator = dataset_iterator
        self.optimizer = optimizer

        # List of param sets
        self.incoming_params = []

        # Dataset stuff
        chunk = next(dataset_iterator)
        self.data_chunk_size = len(chunk)
        self.data_chunk_iterator = iter(chunk)
        self.batch_idx = 0

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

        # List of param sets
        self.incoming_params = []

    # Sync
    def aggr_and_apply_params(self, param_sets):
        # Average and assign params
        for param_id in self.params:
            param_value = 0

            for param_set in param_sets:
                param_value += param_set[param_id]
            
            param_value /= len(param_sets)
            self.params[param_id].assign(param_value)

    # Async
    def apply_params(self, param_set):
        if self.parent is not None:
            # Assign params for relay
            for param_id in self.params:
                self.params[param_id].assign(param_set[param_id].value())
        else:
            # Average params into current (async only works on one set of params at a time)
            for param_id in self.params:
                param_value = (self.params[param_id].value() + param_set[param_id]) / 2
                self.params[param_id].assign(param_value)

    def replace_params(self, param_set):
        for param_id in self.params:
            self.params[param_id].assign(param_set[param_id])

    def get_params(self):
        params = {}
        for param_id in self.params:
            params[param_id] = self.params[param_id].value()
        return params
            
            
class ClusterLite:

    def __init__(self, model_builder, dataset_fn, config):
        self.model_builder = model_builder
        self.dataset_fn = dataset_fn

        self.nodes = {}

        self._parse_config(config)

        self.test_model, self.test_model_params, _, _ = model_builder()

        self._create_nodes()

        self.steps_complete = 0

        self.nsg = NetworkSequenceGenerator(self.node_descs)

        for _ in range(2000):
            self.nsg.generate()

    def _create_nodes(self):

        # Build dataset_iterator
        dataset = self.dataset_fn(self.num_train_samples)
        dataset_iterator = DatasetIterator(dataset, self.batch_size, self.data_chunk_size)

        for node_desc in self.node_descs:

            _, params, forward_pass, build_optimizer = self.model_builder()

            if node_desc['node_type'] == 'ps':
                ps = ParameterServer(node_desc['id'], node_desc['parent'], node_desc['sync_style'], params)
                self.nodes[ps.id] = ps

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

        self.slow_worker_lb = self._get_config_item(config, 'slow_worker_lower_bound_ms')
        self.slow_worker_ub = self._get_config_item(config, 'slow_worker_upper_bound_ms')

        self.node_descs = self._get_config_item(config, 'nodes')

        self.ps_return_threshold = 0 # TODO removed functionality

        self.data_chunk_size = self._get_config_item(config, 'data_chunk_size')

        self.acc_thresholds = self._get_config_item(config, 'acc_thresholds')

        self.record_gantt = self._get_config_item(config, 'record_gantt')
        self.rg_fg = self._get_config_item(config, 'rg_fine_grained')


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

    def process_event(self, event):

        if type(event) == SendParamsEvent:
            params = self.nodes[event.sender_id].get_params()
            self.nodes[event.receiver_id].incoming_params.append(params)

        elif type(event) == ReceiveParamsEvent:
            receiver = self.nodes[event.receiver_id]
            if type(receiver) == Worker:
                # Worker should only have 1 param set in incoming params
                params = receiver.incoming_params[0]
                receiver.incoming_params = []
                receiver.replace_params(params)

        elif type(event) == WorkerStepEvent:
            self.nodes[event.worker_id].train_step()
            self.steps_complete += 1

        elif type(event) == PSApplyEvent:
            ps = self.nodes[event.ps_id]
            if ps.sync_style == 'async':
                params = ps.incoming_params.pop(0)
                ps.apply_params(params)
            elif ps.sync_style == 'sync':
                param_sets = ps.incoming_params
                ps.incoming_params = []
                ps.aggr_and_apply_params(param_sets)

    def run(self):

        x_test, y_test = keras_model.test_dataset(self.num_test_samples)

        while True:

            event = self.nsg.events.pop(0)
            self.process_event(event)

            if self.steps_complete != 0 and self.steps_complete % 100 == 0:
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



from model_and_data_builder import model_builder, dataset_fn

def load_config(config_file_path):
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()
    return config

config = load_config('config.json')

cluster = ClusterLite(model_builder, dataset_fn, config)

cluster.run()