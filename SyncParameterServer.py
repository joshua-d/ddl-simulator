from ParameterServer import ParameterServer
import threading


class SyncParameterServer(ParameterServer):

    # SyncParameterServer does not use return_threshold
    def __init__(self, id, parents, update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold):
        super().__init__(id, parents, update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold)
        

    def run(self):
        
        while True:

            # Get first batch of updates from children
            param_update_buffer = self.ni.ps_wait_for_child_update(self)
            n_updates_received = len(param_update_buffer)

            with self.params_lock:

                while True:

                    # Apply updates
                    for param_update in param_update_buffer:
                        param_update.apply(self.params, self.optimizer)
                        self.n_updates += 1

                    # Break if all children have sent in updates, otherwise wait for more
                    if n_updates_received == len(self.children):
                        break
                    else:
                        param_update_buffer = self.ni.ps_wait_for_child_update(self)
                        n_updates_received += len(param_update_buffer)

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
                    

            # Send params back to children
            for node_id in self.children:
                self.ni.ps_send_to_child(node_id, params)

