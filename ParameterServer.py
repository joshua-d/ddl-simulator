import threading
from Node import Node
from time import perf_counter


class ParameterServer(Node):

    def __init__(self, id, parents, update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold):
        super().__init__(id, parents, update_policies, param_locations, ni)

        # { param_id: tf.Variable }
        self.params = params 
        self.optimizer = optimizer

        self.update_interval = update_interval
        self.n_updates = 0

        self.return_threshold = return_threshold

        # IDs of children
        self.children = []

        # used for get_test_model
        self.params_lock = threading.Lock()

        self.server_thread = threading.Thread(target=self.run, daemon=True)

        # Queue for param updates from children
        self.child_update_queue = []
        self.child_update_queue_cond = threading.Condition()


    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def run(self):
        
        while True:

            # Get first batch of updates from children
            param_update_buffer = self.ni.ps_wait_for_child_update(self)

            first_update_received_time = perf_counter()
            n_updates_received = len(param_update_buffer)
            waiting_nodes = []

            with self.params_lock:

                while True:

                    # Apply updates
                    for param_update in param_update_buffer:
                        param_update.apply(self.params, self.optimizer)
                        waiting_nodes.append(param_update.sender_id)
                        self.n_updates += 1

                    # Wait for more updates, or if threshold has passed, break
                    time_passed = perf_counter() - first_update_received_time
                    if n_updates_received < len(self.children) and time_passed < self.return_threshold:
                        param_update_buffer = self.ni.ps_wait_for_child_update(self, timeout=(self.return_threshold - time_passed))
                        if len(param_update_buffer) == 0:
                            break
                        n_updates_received += len(param_update_buffer)
                    else:
                        break

                params = self.get_params()
                

                # Maybe update parent
                if self.n_updates >= self.update_interval:
                    self.update_parents(gradients=None, param_values=params)
                    self.n_updates = 0

                    # Wait for parents' replacement updates
                    param_update_buffer = self.ni.ps_wait_for_parent_update(self)

                    # Apply updates
                    for param_update in param_update_buffer:
                        param_update.apply(self.params, self.optimizer)

                    params = self.get_params()

                    
            # Send params back to waiting nodes
            if len(waiting_nodes) > 0:
                for node_id in waiting_nodes:
                    self.ni.ps_send_to_child(self.id, node_id, params)


    def start(self):
        self.server_thread.start()
                    
