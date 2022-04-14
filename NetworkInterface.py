from time import sleep
from NetworkEmulator import NetworkEmulator


PARAMS_SIZE = 407_080 * 8 # 407.08 KB
GRADS_SIZE = 407_080 * 8

class NetworkInterface:

    def __init__(self, cluster, bandwidth):
        self.cluster = cluster
        self.nc = NodeCommunication(cluster)
        self.ne = NetworkEmulator(bandwidth)


        # TODO this diagnostic assumes grads and params are same size, and saturation is when there are num_workers msgs at once
        time_per_msg = PARAMS_SIZE / (bandwidth / cluster.num_workers) 
        print('\nNetwork Interface:')
        print('Time per message when network is saturated (%d msgs at once): %f\n' % (cluster.num_workers, time_per_msg))

    def wait_for_params(self, worker):
        return self.nc.wait_for_params(worker)

    def send_gradients(self, wk_id, ps_id, grads):
        # TODO get size of actual grads
        self.ne.send_msg(GRADS_SIZE, lambda: self.nc.send_gradients(wk_id, ps_id, grads))

    def wait_for_grads(self, ps):
        return self.nc.wait_for_grads(ps)

    def send_params(self, wk_id, vals_by_param_id):
        # TODO get size of actual params
        self.ne.send_msg(PARAMS_SIZE, lambda: self.nc.send_params(wk_id, vals_by_param_id))

    def broadcast_params(self, vals_by_param_id):
        for worker in self.cluster.workers:
            self.send_params(worker.id, vals_by_param_id)

    
    def start(self):
        self.ne.start()


# TODO consider removing cluster from NC's fields, pass in through NI
class NodeCommunication:

    def __init__(self, cluster):
        self.cluster = cluster


    def wait_for_params(self, worker):
        with worker.params_queue_cond:
            while len(worker.params_queue) != self.cluster.num_ps:
                worker.params_queue_cond.wait()

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
            while len(ps.grads_queue) == 0:
                ps.grads_queue_cond.wait()

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


    def clear_worker_params_queues(self):
        for worker in self.cluster.workers:
            with worker.params_queue_cond:
                worker.params_queue = []


    def clear_ps_grads_queues(self):
        for ps in self.cluster.parameter_servers.values():
            with ps.grads_queue_cond:
                ps.grads_queue = []
