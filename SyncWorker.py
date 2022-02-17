from Worker import Worker
import threading


# TODO strictly assumes there is only 1 PS

class SyncWorker(Worker):

    def wait_for_and_assign_params(self):
        params_msgs = self.network.wait_for_params(self)
        for vals_by_param_id in params_msgs:
            for param_id in vals_by_param_id:
                self.params[param_id].assign(vals_by_param_id[param_id])

    def train_step(self):
        self.wait_for_and_assign_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        self.send_gradients(gradients)
        

    def start(self):
        self.stop_training = False

        while not self.stop_training:
            self.train_step()
            with self.cluster.steps_completed_lock:
                self.cluster.steps_completed += 1
                if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                    self.stop_training = True  # TODO maybe stop for all workers? trying to throw out in prog, step-schedule seems like a good system
                    break

            