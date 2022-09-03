from threading import Thread, Condition
from time import sleep
import random
from Node import Node, UpdatePolicy, GanttEvent

from MessageTypes import *


class Worker(Node):

    def __init__(self, id, parents, parent_update_policies, param_locations, ni, model_builder, dataset_iterator, optimizer, slow, cluster):
        super().__init__(id, parents, parent_update_policies, param_locations, ni)

        self.model, self.params, self.forward_pass, _ = model_builder()
        self.dataset_iterator = dataset_iterator
        self.optimizer = optimizer
        self.slow = slow

        self.cluster = cluster # TODO don't really want to have to do this, but need it for steps_completed stuff

        # Bool to determine if worker should apply grads to its own cache - if parent has AVERAGE policy
        self.optimize_model_cache = False
        for policy in parent_update_policies.values():
            if policy == UpdatePolicy.AVERAGE:
                self.optimize_model_cache = True

        # Dataset stuff
        chunk = next(dataset_iterator)
        self.data_chunk_size = len(chunk)
        self.data_chunk_iterator = iter(chunk)
        self.batch_idx = 0

        # steps_scheduled decremented only once the gradients for the step are ON the network
        self.steps_scheduled = 0
        self.steps_scheduled_cond = Condition()

        self.steps_completed = 0

        self.working_thread = Thread(target=self.work, daemon=True)

        
    def handle_msg(self, msg):
        # Right now, workers will only receive ReplacementParamsMsgs from their parent PSs 
        if type(msg) == ReplacementParamsMsg:
            self.incoming_parent_msgs.append(msg)

            # If all params are in, wake up working thread
            if len(self.incoming_parent_msgs) == len(self.parents):
                with self.parent_params_ready_cond:
                    self.parent_params_ready = True
                    self.parent_params_ready_cond.notify()
            

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
        self.open_gantt(GanttEvent.WORKER_STEP)
        gradients = self.forward_pass(self.get_next_batch())

        if self.slow:
            sleep(random.randint(self.cluster.slow_worker_lb, self.cluster.slow_worker_ub) / 1000)

        if self.optimize_model_cache:
            self.optimizer.apply_gradients(zip(gradients, self.params.values()))
            params = self.get_params()
        else:
            params = None

        for parent_id in self.parents:
            if self.parent_update_policies[parent_id] == UpdatePolicy.AVERAGE:
                self.ni.send_params_average(self.id, parent_id, params)
            elif self.parent_update_policies[parent_id] == UpdatePolicy.GRADIENT:
                self.ni.send_params_gradient(self.id, parent_id, gradients)

        self._increment_step_counter()

        self.close_gantt()

        self.wait_for_parent_params()


    def work(self):
        while True:
            self.train_step()


    def start(self):
        self.msg_handler_thread.start()
        self.working_thread.start()