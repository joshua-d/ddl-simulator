from ParameterServer import ParameterServer


class SyncParameterServer(ParameterServer):

    def __init__(self, params, optimizer, workers):
        super().__init__(params, optimizer)
        
        self.workers = workers

        self.workers_received = 0
        self.current_grads = []

    
    def on_receive(self, gradients):
        self.current_grads.append(gradients)
        self.workers_received += 1

        if self.workers_received == len(self.workers):
            # All workers have completed their step this round - aggregate!
            print('PS received grads from all workers')

            grads_by_param_id = {}

            for param_id in self.params:
                grads_by_param_id[param_id] = []

            for gradients in self.current_grads:
                for grad, param_id in gradients:
                    grads_by_param_id[param_id].append(grad)

            # average grads for each param
            aggr_grad_by_param_id = {}
            for param_id in self.params:

                for grad_idx in range(len(grads_by_param_id[param_id])):
                    if grad_idx == 0:
                        aggr_grad = grads_by_param_id[param_id][grad_idx]
                    else:
                        aggr_grad += grads_by_param_id[param_id][grad_idx]

                aggr_grad = aggr_grad / len(grads_by_param_id[param_id])
                aggr_grad_by_param_id[param_id] = aggr_grad

            # make apply list and apply grads to params
            apply_list = []
            for param_id in self.params:
                grad = aggr_grad_by_param_id[param_id]
                apply_list.append((grad, self.params[param_id]))

            self.optimizer.apply_gradients(apply_list)

            self.workers_received = 0
            self.current_grads = []

            for worker in self.workers:
                worker.ready_for_next_step_lock.acquire()
                worker.ready_for_next_step = True
                worker.ready_for_next_step_lock.release()