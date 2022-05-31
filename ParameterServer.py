import threading


class ParameterServer:

    def __init__(self, id, params, optimizer, ni):
        # TODO document this!

        self.id = id

        # { param_id: tf.Variable }
        self.params = params
         
        self.optimizer = optimizer

        self.ni = ni

        # TODO update this documentation
        # List of ParamUpdate objects
        # ParamUpdate:
        #
        #   sender_id: id of sender
        #
        #   apply(params, optimizer)
        #       params: ParameterServer.params
        #
        #       applies update
        self.param_update_queue = []
        self.param_update_queue_cond = threading.Condition()

        # used for get_test_model
        self.params_lock = threading.Lock()


    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def start(self):

        # Start by broadcasting params
        self.ni.broadcast_params(self.get_params())
        
        while True:
            param_update_buffer = self.ni.ps_wait_for_update(self)

            waiting_nodes = []

            # Apply all updates
            with self.params_lock:
                for param_update in param_update_buffer:
                    param_update.apply(self.params, self.optimizer)
                    if param_update.return_params:
                        waiting_nodes.append(param_update.sender_id)
                    
            # Send params back to waiting nodes
            if len(waiting_nodes) > 0:
                vals_by_param_id = self.get_params() # TODO may need lock on here because cluster and self reading at same time?
                for node_id in waiting_nodes:
                    self.ni.send_params(node_id, vals_by_param_id)
                    
