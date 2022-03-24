from time import sleep


class NodeCommunication:

    def __init__(self, cluster):
        self.cluster = cluster


    def wait_for_params(self, worker):
        with worker.params_queue_cond:
            worker.params_queue_cond.wait_for(lambda: len(worker.params_queue) == self.cluster.num_ps)

            # All params are in, move them out of queue and return to worker
            params_msgs = worker.params_queue
            worker.params_queue = []

        return params_msgs


    def send_gradients(self, wk_id, ps_id, grads):
        ps = self.cluster.parameter_servers[ps_id]

        # Place grad & wk_id in grads queue and notify PS
        with ps.grads_queue_cond:
            ps.grads_queue.append((grads, wk_id))
            ps.grads_queue_cond.notify()


    def wait_for_grads(self, ps):

        # Wait for worker to request params or send grads
        with ps.grads_queue_cond:
            ps.grads_queue_cond.wait_for(len(ps.grads_queue) > 0)

            # A request has come in, move everything out of queues and return to ps
            grads_queue_buffer = ps.grads_queue
            ps.grads_queue = []
        
        return grads_queue_buffer


    def send_params(self, wk_id, vals_by_param_id):
        worker = self.cluster.workers[wk_id]

        # Place params in param queue and notify worker
        with worker.params_queue_cond:
            worker.params_queue.append(vals_by_param_id)
            worker.params_queue_cond.notify()


    def broadcast_params(self, vals_by_param_id):
        for worker in self.cluster.workers:
            self.send_params(worker.id, vals_by_param_id)


    def flush_worker_params_queues(self):
        for worker in self.cluster.workers:
            with worker.params_queue_cond:
                worker.params_queue = []
