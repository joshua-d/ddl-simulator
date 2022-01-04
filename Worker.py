

from tensorflow._api.v2 import data


class Worker:

    def __init__(self, cluster, model_builder, dataset_iterator, param_locations):
        self.cluster = cluster
        self.model, self.params, self.forward_pass = model_builder()
        self.dataset_iterator = dataset_iterator
        self.param_locations = param_locations

    def request_params(self):
        for ps_id in self.param_locations:
            params = self.cluster.parameter_servers[ps_id].on_request()
            for param_id in params:
                self.params[param_id].assign(params[param_id])


    def send_gradients(self, gradients):
        for ps_id in self.param_locations:
            send_list = []
            for param_id in self.param_locations[ps_id]:
                send_list.append((gradients[param_id], param_id))
            self.cluster.parameter_servers[ps_id].on_receive(send_list)

    def train(self):
        self.request_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        self.send_gradients(gradients)