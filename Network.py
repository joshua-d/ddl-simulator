

class Network:

    def __init__(self, cluster):
        self.cluster = cluster

    def request_params_and_wait(self, worker):
        for ps_id in self.cluster.parameter_servers:
            ps = self.cluster.parameter_servers[ps_id]

            # Place worker in waiting workers queue and notify PS
            with ps.grads_queue_cond:
                ps.waiting_workers.append(worker.id)
                ps.grads_queue_cond.notify()
        
        # Wait for all PSs to send params back
        with worker.params_queue_cond:
            worker.params_queue_cond.wait_for(lambda: len(worker.params_queue) == self.cluster.num_ps)

            # All params are in, move them out of queue and return to worker
            params_msgs = worker.params_queue
            worker.params_queue = []
        
        return params_msgs


    def send_gradients(self, ps_id, grads):
        ps = self.cluster.parameter_servers[ps_id]

        # Place grad in grads queue and notify PS
        with ps.grads_queue_cond:
            ps.grads_queue.append(grads)
            ps.grads_queue_cond.notify()


    def wait_for_worker_request(self, ps):

        # Wait for worker to request params or send grads
        with ps.grads_queue_cond:
            ps.grads_queue_cond.wait_for(lambda: len(ps.waiting_workers) > 0 or len(ps.grads_queue) > 0)

            # A request has come in, move everything out of queues and return to ps

            grads_queue_buffer = ps.grads_queue
            ps.grads_queue = []
            
            waiting_workers_buffer = ps.waiting_workers
            ps.waiting_workers = []

        return grads_queue_buffer, waiting_workers_buffer


    def send_params(self, wk_id, vals_by_param_id):
        worker = self.cluster.workers[wk_id]

        # Place params in param queue and notify worker
        with worker.params_queue_cond:
            worker.params_queue.append(vals_by_param_id)
            worker.params_queue_cond.notify()
