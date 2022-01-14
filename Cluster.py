import tensorflow as tf
import threading

from ParameterServer import ParameterServer


# For now assuming cluster is the outermost name for the system, i. e. only one cluster per simulator run
class Cluster:

    def __init__(self, model_builder, config):
        self.model_builder = model_builder
        self._create_parameter_servers(config)

        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_lock = threading.Lock()


    def _create_parameter_servers(self, config):
        num_ps = self._check_config_item(config, 'num_ps')
        learning_rate = self._check_config_item(config, 'learning_rate')
        
        _, params, _ = self.model_builder()

        # round robin placement
        params_objs = []
        for i in range(num_ps):
            params_objs.append({})

        next_ps_ind = 0
        for param_id in params:
            params_objs[next_ps_ind][param_id] = params[param_id]
            next_ps_ind = (next_ps_ind + 1) % num_ps

        self.parameter_servers = {}
        for i in range(num_ps):
            self.parameter_servers['ps%d' % i] = ParameterServer(params_objs[i], tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

    # TODO make static
    def _check_config_item(self, config, item):
        if config[item] is None:
            raise Exception('%s not in config' % item)
        else:
            return config[item]
