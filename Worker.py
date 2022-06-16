import threading
from time import sleep
import random
from Node import Node


class Worker(Node):

    def __init__(self, id, parents, update_policies, param_locations, ni, model_builder, dataset_iterator, cluster):
        super().__init__(id, parents, update_policies, param_locations, ni)

        self.model, self.params, self.forward_pass, _ = model_builder()
        self.dataset_iterator = dataset_iterator

        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        # steps_scheduled decremented only once the gradients for the step are ON the network
        self.steps_scheduled = 0
        self.steps_scheduled_cond = threading.Condition()

        self.steps_completed = 0

        self.working_thread = threading.Thread(target=self.work, daemon=True)


    def _increment_step_counter(self):
        # Increment steps completed
        with self.cluster.steps_completed_cond:
            self.cluster.steps_completed += 1
            self.steps_completed += 1
            if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                self.cluster.steps_completed_cond.notify()


    def wait_for_params(self):
        param_updates = self.ni.worker_wait_for_params(self)

        for param_update in param_updates:
            param_update.apply(params=self.params, optimizer=None)


    def train_step(self):
        gradients = self.forward_pass(next(self.dataset_iterator))

        if self.id < self.cluster.num_slow_workers:
            sleep(random.randint(self.cluster.slow_worker_lb, self.cluster.slow_worker_ub) / 1000)

        # print("Worker %d updating parent" % self.id)
        self.update_parents(gradients, None)

        self._increment_step_counter()

        # print("Worker %d waiting for params" % self.id)
        self.wait_for_params()


    def work(self):
        while True:
            self.train_step()


    def start(self):
        self.working_thread.start()