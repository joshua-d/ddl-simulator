import threading
from time import sleep
import random
from Node import Node, UpdatePolicy


class Worker(Node):

    def __init__(self, id, parents, update_policies, param_locations, ni, model_builder, dataset_iterator, optimizer, cluster):
        super().__init__(id, parents, update_policies, param_locations, ni)

        self.model, self.params, self.forward_pass, _ = model_builder()
        self.dataset_iterator = dataset_iterator
        self.optimizer = optimizer

        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        # Bool to determine if worker should apply grads to its own cache - if parent has AVERAGE policy
        self.optimize_model_cache = False
        for policy in update_policies.values():
            if policy == UpdatePolicy.AVERAGE:
                self.optimize_model_cache = True

        # Dataset stuff
        chunk = next(dataset_iterator)
        self.data_chunk_size = len(chunk)
        self.data_chunk_iterator = iter(chunk)
        self.batch_idx = 0

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


    # TODO Identical to PS - rename these?
    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def wait_for_params(self):
        param_updates = self.ni.worker_wait_for_params(self)

        for param_update in param_updates:
            param_update.apply(params=self.params, optimizer=None)


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

        if self.id < self.cluster.num_slow_workers:
            sleep(random.randint(self.cluster.slow_worker_lb, self.cluster.slow_worker_ub) / 1000)

        if self.optimize_model_cache:
            self.optimizer.apply_gradients(zip(gradients, self.params.values()))
            params = self.get_params()
        else:
            params = None

        self.update_parents(gradients, params)

        self._increment_step_counter()

        self.wait_for_params()


    def work(self):
        while True:
            self.train_step()


    def start(self):
        self.working_thread.start()