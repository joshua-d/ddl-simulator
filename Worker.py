from tensorflow._api.v2 import data
import threading
from time import sleep
import random


class Worker:

    def __init__(self, id, model_builder, dataset_iterator, param_locations, ni, cluster):
        self.id = id
        self.model, self.params, self.forward_pass, _ = model_builder()
        self.dataset_iterator = dataset_iterator

        # { param_id: ps_id }
        self.param_locations = param_locations

        self.ni = ni
        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        self.param_update_queue = []
        self.param_update_queue_cond = threading.Condition()

        # steps_scheduled decremented only once the gradients for the step are ON the network
        self.steps_scheduled = 0
        self.steps_scheduled_cond = threading.Condition()

        self.steps_completed = 0


    def wait_for_params(self):
        param_updates = self.ni.worker_wait_for_params(self)

        for param_update in param_updates:
            param_update.apply(self.params, None)


    # gradients: { param id: param gradient }, returned from forward_pass
    def send_gradients(self, gradients):

        for ps_id in self.param_locations:
            send_list = []
            for param_id in self.param_locations[ps_id]:
                send_list.append((gradients[param_id], param_id))
            
            self.ni.worker_send_gradients(self.id, ps_id, send_list)


    def train_step(self):
        self.wait_for_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        if self.id < self.cluster.num_slow_workers:
            sleep(random.randint(self.cluster.slow_worker_lb, self.cluster.slow_worker_ub) / 1000)
        self.send_gradients(gradients)


    def start(self):
        while True:

            # Do step
            self.train_step()

            # Increment steps completed
            with self.cluster.steps_completed_cond:
                self.cluster.steps_completed += 1
                self.steps_completed += 1
                if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                    self.cluster.steps_completed_cond.notify()
