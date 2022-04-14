from tensorflow._api.v2 import data
import threading
from time import sleep
import random


class Worker:

    def __init__(self, id, model_builder, dataset_iterator, param_locations, ni, cluster):
        self.id = id
        self.model, self.params, self.forward_pass = model_builder()
        self.dataset_iterator = dataset_iterator

        # { param_id: ps_id }
        self.param_locations = param_locations

        self.ni = ni
        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        self.params_queue = []
        self.params_queue_cond = threading.Condition()

        self.stop = False

        # steps_scheduled decremented only once the gradients for the step are ON the network
        self.steps_scheduled = 0
        self.steps_scheduled_cond = threading.Condition()


    def wait_for_params(self): # TODO consider renaming params_msgs here and in Network
        params_msgs = self.ni.wait_for_params(self)

        for vals_by_param_id in params_msgs:
            for param_id in vals_by_param_id:
                self.params[param_id].assign(vals_by_param_id[param_id])


    def send_gradients(self, gradients):

        for ps_id in self.param_locations:
            send_list = []
            for param_id in self.param_locations[ps_id]:
                send_list.append((gradients[param_id], param_id))
            
            self.ni.send_gradients(self.id, ps_id, send_list)


    def train_step(self):
        self.wait_for_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        if self.id == 0 or self.id == 1:
            sleep(random.randint(20, 40) / 1000) # 20ms - 40ms
        self.send_gradients(gradients)


    def start(self):
        self.stop = False

        while not self.stop:

            # Wait until there are steps scheduled
            with self.steps_scheduled_cond:
                while self.steps_scheduled == 0:
                    self.steps_scheduled_cond.wait()

            # Do step
            self.train_step()

            # Decrement steps scheduled
            with self.steps_scheduled_cond: 
                self.steps_scheduled -= 1
                if self.steps_scheduled == 0:
                    self.steps_scheduled_cond.notify_all()
