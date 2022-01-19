import tensorflow as tf
import threading

from ParameterServer import ParameterServer
from Worker import Worker


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

        self._create_parameter_servers(config)
        self._create_workers(config)

        self.test_model, self.test_model_params, _ = model_builder()

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_lock = threading.Lock()


    def _create_parameter_servers(self, config):
        num_ps = self._check_config_item(config, 'num_ps')
        learning_rate = self._check_config_item(config, 'learning_rate')

        # { PS_id -> ParameterServer }
        self.parameter_servers = {}

        # { PS_id -> [v1_id, v2_id, ...] }
        self.param_locations = {}

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
            self.parameter_servers[ps_id] = ParameterServer(params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=learning_rate))
            self.param_locations[ps_id] = list(params_objs[i].keys())

    
    def _create_workers(self, config):
        num_workers = self._check_config_item(config, 'num_workers')
        
        # TODO dict for consistency with self.parameter_servers
        # [ Worker ]
        self.workers = []

        for i in range(num_workers):
            dataset_iterator = iter(self.dataset_fn(i))
            self.workers.append(Worker(self, self.model_builder, dataset_iterator))


    def get_test_model(self):
        for ps_id in self.param_locations:
            ps = self.parameter_servers[ps_id]
            ps.params_lock.acquire()
            params = ps.on_request()
            for param_id in params:
                self.test_model_params[param_id].assign(params[param_id])
            ps.params_lock.release()

        return self.test_model


    # TODO make static, maybe rename to get
    def _check_config_item(self, config, item):
        if config[item] is None:
            raise Exception('%s not in config' % item)
        else:
            return config[item]
