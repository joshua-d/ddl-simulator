from tensorflow._api.v2 import data
import threading
from time import sleep
import random


class Worker:

    def __init__(self, id, model_builder, dataset_iterator, param_locations, ni, cluster):
        self.id = id
        self.model, self.params, self.forward_pass, build_optimizer = model_builder()
        self.dataset_iterator = dataset_iterator

        # { param_id: ps_id }
        self.param_locations = param_locations

        self.ni = ni
        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        self.params_queue = []
        self.params_queue_cond = threading.Condition()

        # steps_scheduled decremented only once the gradients for the step are ON the network
        self.steps_scheduled = 0
        self.steps_scheduled_cond = threading.Condition()

        self.steps_completed = 0
        
        # tus-idea-a worker needs optimizer
        self.optimizer = build_optimizer(cluster.learning_rate)


    def wait_for_params(self): # TODO consider renaming params_msgs here and in Network
        params_msgs = self.ni.wait_for_params(self)

        for vals_by_param_id in params_msgs:
            for param_id in vals_by_param_id:
                self.params[param_id].assign(vals_by_param_id[param_id])


    # tus-idea-a Only works for 1 PS
    def train_step(self):

        self.wait_for_params()
        send_list = []

        num_steps_this_round = 0

        # If slow worker, perform 1 to S steps before sending
        if self.id < self.cluster.num_slow_workers:
            steps = random.randint(1, self.cluster.S)
        else:
            steps = 1

        for _ in range(steps):
            gradients = self.forward_pass(next(self.dataset_iterator))
            
            apply_list = []
            grads_list = []
            for param_id in self.params:
                apply_list.append((gradients[param_id], self.params[param_id]))
                grads_list.append((gradients[param_id], param_id))

            send_list.append(grads_list)
            self.optimizer.apply_gradients(apply_list)
            num_steps_this_round += 1

        self.ni.send_gradients(self.id, 'ps0', send_list)

        return num_steps_this_round
        


    def start(self):
        while True:

            # Do step
            num_steps_this_round = self.train_step()

            # Increment steps completed
            with self.cluster.steps_completed_cond:
                self.steps_completed += 1
                self.cluster.steps_completed += num_steps_this_round
                if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                    self.cluster.steps_completed_cond.notify()
