from time import sleep
from NetworkEmulator import NetworkEmulator

from MessageTypes import *

class NetworkInterface:

    def __init__(self, cluster, node_bws, PARAMS_SIZE, GRADS_SIZE):
        self.cluster = cluster # TODO so far, just need this in order to access node objects for NC
        self.PARAMS_SIZE = PARAMS_SIZE # size of a params msg
        self.GRADS_SIZE = GRADS_SIZE # size of a grads msg
        
        # Generate ID guides for Gantt
        worker_ids = []
        mid_lvl_ps_ids = []
        for node_desc in cluster.node_descs:
            if node_desc['node_type'] == 'worker':
                worker_ids.append(node_desc['id'])
            elif node_desc['id'] != 0 and node_desc['node_type'] == 'ps':
                mid_lvl_ps_ids.append(node_desc['id'])


        self.nc = NodeCommunication(cluster)
        self.ne = NetworkEmulator(node_bws, worker_ids, mid_lvl_ps_ids, cluster.nodes, cluster.record_gantt, cluster.rg_fg)

        print('Params size: %d' % PARAMS_SIZE)
        print('Grads size: %d' % GRADS_SIZE)


    def ps_send_to_child(self, from_id, to_id, vals_by_param_id):
        # TODO get actual size dynamically
        self.ne.send_msg(from_id, to_id, self.PARAMS_SIZE, lambda: self.nc.ps_send_to_child(to_id, vals_by_param_id))

    def send_params_gradient(self, from_id, to_id, grads_by_param_id):
        # TODO get actual size dynamically
        self.ne.send_msg(from_id, to_id, self.GRADS_SIZE, lambda: self.nc.send_params_gradient(from_id, to_id, grads_by_param_id))

    def send_params_average(self, from_id, to_id, vals_by_param_id):
        # TODO get actual size dynamically
        self.ne.send_msg(from_id, to_id, self.PARAMS_SIZE, lambda: self.nc.send_params_average(from_id, to_id, vals_by_param_id))

    
    def start(self):
        self.ne.start()


class NetworkInterfaceBypass:

    def __init__(self, cluster):
        self.cluster = cluster
        self.nc = NodeCommunication(cluster)

        print('BYPASSING NETWORK INTERFACE')


    def ps_send_to_child(self, from_id, to_id, vals_by_param_id):
        self.nc.ps_send_to_child(to_id, vals_by_param_id)

    def send_params_gradient(self, from_id, to_id, grads_by_param_id):
        self.nc.send_params_gradient(from_id, to_id, grads_by_param_id)

    def send_params_average(self, from_id, to_id, vals_by_param_id):
        self.nc.send_params_average(from_id, to_id, vals_by_param_id)

    

    def start(self):
        pass


# TODO consider removing cluster from NC's fields, pass in through NI
class NodeCommunication:

    def __init__(self, cluster):
        self.cluster = cluster


    # Used by a PS to place a ReplacementParamsMsg in a child's msg queue
    def ps_send_to_child(self, node_id, vals_by_param_id):
        node = self.cluster.nodes[node_id]

        # Build ReplacementParamsMsg
        params_msg = ReplacementParamsMsg(vals_by_param_id)

        # Place msg in queue and notify node
        with node.msg_queue_cond:
            node.msg_queue.append(params_msg)
            node.msg_queue_cond.notify()


    # TODO rename this? these?
    # Used by a child to place a GradientsMsg in a parent PS's msg queue
    def send_params_gradient(self, from_id, to_id, grads_by_param_id):
        node = self.cluster.nodes[to_id]

        # Build GradientsMsg
        grads_msg = GradientsMsg(grads_by_param_id, from_id)

        # Place msg in queue and notify node
        with node.msg_queue_cond:
            node.msg_queue.append(grads_msg)
            node.msg_queue_cond.notify()


    # Used by a child to place a ParamsMsg in a parent PS's msg queue
    def send_params_average(self, from_id, to_id, vals_by_param_id):
        node = self.cluster.nodes[to_id]

        # Build ParamsMsg
        params_msg = ParamsMsg(vals_by_param_id, from_id)

        # Place msg in queue and notify node
        with node.msg_queue_cond:
            node.msg_queue.append(params_msg)
            node.msg_queue_cond.notify()
