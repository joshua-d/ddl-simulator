

from tensorflow._api.v2 import data


class Worker:

    def __init__(self, cluster, model_builder, dataset_iterator, param_locations):
        self.cluster = cluster
        self.model, self.params, self.forward_pass = model_builder()
        self.dataset_iterator = dataset_iterator
        self.param_locations = param_locations

        self.num_steps_completed = 0
        self.stop_training = False

    def request_params(self):
        for ps_id in self.param_locations:
            ps = self.cluster.parameter_servers[ps_id]
            ps.params_lock.acquire() # TODO - this locking logic can be done better, should also encapsulate it in ps interface
            params = ps.on_request()
            for param_id in params:
                self.params[param_id].assign(params[param_id])
            ps.params_lock.release()


    def send_gradients(self, gradients):
        for ps_id in self.param_locations:
            send_list = []
            for param_id in self.param_locations[ps_id]:
                send_list.append((gradients[param_id], param_id))
            
            ps = self.cluster.parameter_servers[ps_id]
            ps.params_lock.acquire()
            ps.on_receive(send_list)
            ps.params_lock.release()


    def train_step(self):
        self.request_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        self.send_gradients(gradients)

    def train(self):
        self.stop_training = False
        while not self.stop_training:
            self.train_step()
            self.cluster.steps_completed_lock.acquire()
            self.cluster.steps_completed += 1
            if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                self.stop_training = True  # TODO maybe stop for all workers? trying to throw out in prog step - schedule seems like a good system
            self.cluster.steps_completed_lock.release()