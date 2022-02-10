from ParameterServer import ParameterServer
import threading


# TODO this is currently built to have its functions called by a worker thread - eventually,
# its own thread would probably be better
class SyncParameterServer(ParameterServer):

    def __init__(self, params, optimizer, workers, cluster):
        super().__init__(params, optimizer)
        
        self.workers = workers

        self.workers_received = 0
        self.current_grads = []

        self.sync_cond = threading.Condition()

        self.cluster = cluster

        # TODO debug
        self.print_lock = threading.Lock()

    
    # TODO may make more sense to have parallelism handling in SyncWorker - same function though
    # TODO in the future, params_lock for PS and sync_cond for SyncPS could be same thing because condition can function as a lock
    # This fn now holds the logic for counting steps and stopping training
    def on_receive(self, gradients):
        with self.sync_cond:

            self.current_grads.append(gradients)
            self.workers_received += 1

            if self.workers_received == len(self.workers):
                # All workers have completed their step this round - apply in order received

                # with self.print_lock:
                #     print('PS received grads from all workers - applying grads', flush=True)

                for grads_list in self.current_grads:
                    apply_list = []
                    for grad, param_id in grads_list:
                        apply_list.append((grad, self.params[param_id]))
                    self.optimizer.apply_gradients(apply_list)

                # print('Done.')

                self.workers_received = 0
                self.current_grads = []

                self.cluster.steps_completed += len(self.workers)
                if self.cluster.steps_completed >= self.cluster.steps_scheduled:
                    for worker in self.workers:
                        worker.stop_training = True

                self.sync_cond.notify_all()
                
            else:
                self.sync_cond.wait()


    def apply_current_and_reset(self):
        for grads_list in self.current_grads:
            apply_list = []
            for grad, param_id in grads_list:
                apply_list.append((grad, self.params[param_id]))
            self.optimizer.apply_gradients(apply_list)

        self.workers_received = 0
        self.current_grads = []