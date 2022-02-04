from Worker import Worker
import threading

class SyncWorker(Worker):

    def __init__(self, cluster, id, model_builder, dataset_iterator):
        super().__init__(cluster, id, model_builder, dataset_iterator)
        self.ready_for_next_step = True

        self.ready_for_next_step_lock = threading.Lock()

    def train_step(self):
        self.request_params()
        gradients = self.forward_pass(next(self.dataset_iterator))
        self.send_gradients(gradients)

    def train(self):
        self.stop_training = False
        while not self.stop_training:
            if self.ready_for_next_step:
                self.ready_for_next_step_lock.acquire()
                self.ready_for_next_step = False
                self.ready_for_next_step_lock.release()

                self.train_step()
                print('Worker %d completed step ' % (self.id))

                self.cluster.steps_completed_lock.acquire()
                self.cluster.steps_completed += 1
                if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                    self.stop_training = True  # TODO maybe stop for all workers? trying to throw out in prog step - schedule seems like a good system
                self.cluster.steps_completed_lock.release()