from ParameterServer import ParameterServer
import threading


# TODO Strictly assumes there is only 1 PS

class SyncParameterServer(ParameterServer):

    def __init__(self, id, parents, update_policies, param_locations, ni, params, optimizer, update_interval):
        super().__init__(id, parents, update_policies, param_locations, ni, params, optimizer, update_interval)

        self.round_num_updates_received = 0


    def reset_round(self):
        self.round_num_updates_received = 0
        

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
                    self.round_num_updates_received += 1

                vals_by_param_id = self.get_params()

            # Maybe update parent
            if self.n_updates == self.update_interval:
                self.update_parents(gradients=None, param_values=vals_by_param_id)
                self.n_updates = 0
                    
            # If all children have sent updates, send params back to waiting nodes
            if self.round_num_updates_received == len(self.children):
                self.round_num_updates_received = 0
                for node_id in waiting_nodes:
                    self.ni.send_params_replace(node_id, vals_by_param_id)

