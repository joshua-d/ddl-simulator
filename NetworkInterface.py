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

    def ps_wait_for_child_update(self, ps, timeout=None):
        return self.nc.ps_wait_for_child_update(ps, timeout)

    def ps_wait_for_parent_update(self, ps, timeout=None):
        return self.nc.ps_wait_for_parent_update(ps, timeout)


    def ps_send_to_child(self, node_id, vals_by_param_id):
        # TODO get actual size dynamically
        self.ne.send_msg(self.PARAMS_SIZE, lambda: self.nc.ps_send_to_child(node_id, vals_by_param_id))

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

    def ps_wait_for_child_update(self, ps, timeout=None):
        return self.nc.ps_wait_for_child_update(ps, timeout)

    def ps_wait_for_parent_update(self, ps, timeout=None):
        return self.nc.ps_wait_for_parent_update(ps, timeout)


    def ps_send_to_child(self, node_id, vals_by_param_id):
        self.nc.ps_send_to_child(node_id, vals_by_param_id)

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


    # Waits on worker.parent_update_queue_cond for enough ReplacementParamUpdates to update all of worker's params
    def worker_wait_for_params(self, worker):
        with worker.parent_update_queue_cond:

            while len(worker.parent_update_queue) != len(worker.parents):
                worker.parent_update_queue_cond.wait()

            # All ReplacementParamUpdates are in, move them out of queue and return to worker
            param_updates = worker.parent_update_queue
            worker.parent_update_queue = []

        return param_updates


    # Waits on ps.child_update_queue_cond for a ParamUpdate to enter the queue
    def ps_wait_for_child_update(self, ps, timeout=None):

        # Wait for a param update to come in
        with ps.child_update_queue_cond:
            ps.child_update_queue_cond.wait_for(lambda: len(ps.child_update_queue) != 0, timeout=timeout)

            # A param update has come in, move everything out of queue and return to ps
            child_update_queue_buffer = ps.child_update_queue
            ps.child_update_queue = []
        
        return child_update_queue_buffer

    
    # Waits on ps.parent_update_queue_cond for a ParamUpdate to enter the queue
    def ps_wait_for_parent_update(self, ps, timeout=None):

        # Wait for a param update to come in
        with ps.parent_update_queue_cond:
            ps.parent_update_queue_cond.wait_for(lambda: len(ps.parent_update_queue) != 0, timeout=timeout)

            # A param update has come in, move everything out of queue and return to ps
            parent_update_queue_buffer = ps.parent_update_queue
            ps.parent_update_queue = []
        
        return parent_update_queue_buffer


    # Places a ReplacementParamUpdate in a node's parent_update_queue
    def ps_send_to_child(self, node_id, vals_by_param_id):
        node = self.cluster.nodes[node_id]

        # Build ReplacementParamUpdate
        param_update = ReplacementParamUpdate(vals_by_param_id)

        # Place param update in queue and notify node
        with node.parent_update_queue_cond:
            node.parent_update_queue.append(param_update)
            node.parent_update_queue_cond.notify()


    # Places a GradientParamUpdate in a PS's child_update_queue
    def send_params_gradient(self, node_id, grads_by_param_id, sender_id):
        node = self.cluster.nodes[node_id]

        # Build GradientParamUpdate
        param_update = GradientParamUpdate(grads_by_param_id, sender_id)

        # Place param update in queue and notify node
        with node.child_update_queue_cond:
            node.child_update_queue.append(param_update)
            node.child_update_queue_cond.notify()


    # Places a AverageParamUpdate in a PS's child_update_queue
    def send_params_average(self, node_id, vals_by_param_id, sender_id):
        node = self.cluster.nodes[node_id]

        # Build AverageParamUpdate
        param_update = AverageParamUpdate(vals_by_param_id, sender_id)

        # Place param update in queue and notify node
        with node.child_update_queue_cond:
            node.child_update_queue.append(param_update)
            node.child_update_queue_cond.notify()
