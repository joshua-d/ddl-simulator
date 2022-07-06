import threading
from Node import Node


class ParameterServer(Node):

    def __init__(self, id, parents, update_policies, param_locations, ni, params, optimizer, update_interval):
        super().__init__(id, parents, update_policies, param_locations, ni)

        # { param_id: tf.Variable }
        self.params = params 
        self.optimizer = optimizer

        self.update_interval = update_interval
        self.n_updates = 0

        # IDs of children
        self.children = []

        # used for get_test_model
        self.params_lock = threading.Lock()

        self.server_thread = threading.Thread(target=self.run, daemon=True)


    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def run(self):
        
        while True:
            param_update_buffer = self.ni.ps_wait_for_update(self)

            waiting_nodes = []

            # Apply all updates
            with self.params_lock:
                for param_update in param_update_buffer:
                    param_update.apply(self.params, self.optimizer)
                    if param_update.return_params:
                        waiting_nodes.append(param_update.sender_id)
                        self.n_updates += 1

                vals_by_param_id = self.get_params()

            # Maybe update parent
            if self.n_updates == self.update_interval:
                self.update_parents(gradients=None, param_values=vals_by_param_id)
                self.n_updates = 0
                    
            # Send params back to waiting nodes
            if len(waiting_nodes) > 0:
                for node_id in waiting_nodes:
                    self.ni.send_params_replace(node_id, vals_by_param_id)


    def start(self):
        self.server_thread.start()
                    
