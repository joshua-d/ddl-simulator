from time import sleep
from NetworkEmulator import NetworkEmulator
from GradientParamUpdate import GradientParamUpdate
from ReplacementParamUpdate import ReplacementParamUpdate
from AverageParamUpdate import AverageParamUpdate

class NetworkInterface:

    def __init__(self, cluster, bandwidth, PARAMS_SIZE, GRADS_SIZE):
        self.cluster = cluster
        self.PARAMS_SIZE = PARAMS_SIZE # size of a params msg
        self.GRADS_SIZE = GRADS_SIZE # size of a grads msg
        
        self.nc = NodeCommunication(cluster)
        self.ne = NetworkEmulator(bandwidth)


        # TODO this diagnostic assumes grads and params are same size, and saturation is when there are num_workers msgs at once
        print('\nNetwork Interface:')
        print('Bandwidth: %s' % '{:,}'.format(bandwidth))
        
        if self.cluster.base_msg_time != 0:
            print("Time to send 1 msg: %f" % self.cluster.base_msg_time)

        sat_time = PARAMS_SIZE / (bandwidth / cluster.num_workers) 
        print('Time per message when network is saturated (%d msgs at once): %f\n' % (cluster.num_workers, sat_time))


    def worker_wait_for_params(self, worker):
        return self.nc.worker_wait_for_params(worker)

    def ps_wait_for_update(self, ps):
        return self.nc.ps_wait_for_update(ps)


    def send_params_replace(self, node_id, vals_by_param_id):
        # TODO get actual size dynamically
        self.ne.send_msg(self.PARAMS_SIZE, lambda: self.nc.send_params_replace(node_id, vals_by_param_id))

    def send_params_gradient(self, node_id, grads_by_param_id, sender_id):
        # TODO get actual size dynamically
        self.ne.send_msg(self.GRADS_SIZE, lambda: self.nc.send_params_gradient(node_id, grads_by_param_id, sender_id))

    def send_params_average(self, node_id, vals_by_param_id, sender_id):
        # TODO get actual size dynamically
        self.ne.send_msg(self.PARAMS_SIZE, lambda: self.nc.send_params_average(node_id, vals_by_param_id, sender_id))

    
    def start(self):
        self.ne.start()


class NetworkInterfaceBypass:

    def __init__(self, cluster):
        self.cluster = cluster
        self.nc = NodeCommunication(cluster)

        print('BYPASSING NETWORK INTERFACE')


    def worker_wait_for_params(self, worker):
        return self.nc.worker_wait_for_params(worker)

    def ps_wait_for_update(self, ps):
        return self.nc.ps_wait_for_update(ps)


    def send_params_replace(self, node_id, vals_by_param_id):
        self.nc.send_params_replace(node_id, vals_by_param_id)

    def send_params_gradient(self, node_id, grads_by_param_id, sender_id):
        self.nc.send_params_gradient(node_id, grads_by_param_id, sender_id)

    def send_params_average(self, node_id, vals_by_param_id, sender_id):
        self.nc.send_params_average(node_id, vals_by_param_id, sender_id)

    

    def start(self):
        pass


# TODO consider removing cluster from NC's fields, pass in through NI
class NodeCommunication:

    def __init__(self, cluster):
        self.cluster = cluster


    # Waits on worker.param_update_queue_cond for enough ReplacementParamUpdates to update all of worker's params
    def worker_wait_for_params(self, worker):
        with worker.param_update_queue_cond:

            while len(worker.param_update_queue) != len(worker.parents):
                worker.param_update_queue_cond.wait()

            # All ReplacementParamUpdates are in, move them out of queue and return to worker
            param_updates = worker.param_update_queue
            worker.param_update_queue = []

        return param_updates


    # Waits on ps.param_update_queue_cond for a ParamUpdate to enter the queue
    def ps_wait_for_update(self, ps):

        # Wait for a param update to come in
        with ps.param_update_queue_cond:
            while len(ps.param_update_queue) == 0:
                ps.param_update_queue_cond.wait()

            # A param update has come in, move everything out of queue and return to ps
            param_update_queue_buffer = ps.param_update_queue
            ps.param_update_queue = []
        
        return param_update_queue_buffer


    # Sends a ReplacementParamUpdate to a node
    def send_params_replace(self, node_id, vals_by_param_id):
        node = self.cluster.nodes[node_id]

        # Build ReplacementParamUpdate
        param_update = ReplacementParamUpdate(vals_by_param_id)

        # Place param update in queue and notify node
        with node.param_update_queue_cond:
            node.param_update_queue.append(param_update)
            node.param_update_queue_cond.notify()


    def send_params_gradient(self, node_id, grads_by_param_id, sender_id):
        node = self.cluster.nodes[node_id]

        # Build GradientParamUpdate
        param_update = GradientParamUpdate(grads_by_param_id, sender_id)

        # Place param update in queue and notify node
        with node.param_update_queue_cond:
            node.param_update_queue.append(param_update)
            node.param_update_queue_cond.notify()


    def send_params_average(self, node_id, vals_by_param_id, sender_id):
        node = self.cluster.nodes[node_id]

        # Build AverageParamUpdate
        param_update = AverageParamUpdate(vals_by_param_id, sender_id)

        # Place param update in queue and notify node
        with node.param_update_queue_cond:
            node.param_update_queue.append(param_update)
            node.param_update_queue_cond.notify()


    def clear_worker_param_update_queues(self):
        for worker in self.cluster.workers:
            with worker.param_update_queue_cond:
                worker.param_update_queue = []


    def clear_ps_param_update_queues(self):
        for ps in self.cluster.parameter_servers:
            with ps.param_update_queue_cond:
                ps.param_update_queue = []
