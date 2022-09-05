from AsyncParameterServer import AsyncParameterServer
from Node import GanttEvent, UpdatePolicy
from threading import Condition

from MessageTypes import *


class SyncParameterServer(AsyncParameterServer):

    # SyncParameterServer does not use return_threshold
    def __init__(self, id, parents, update_policy, parent_update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold):
        super().__init__(id, parents, update_policy, parent_update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold)
        
        self.child_params_ready = False
        self.child_params_ready_cond = Condition()


    def handle_msg(self, msg):
        if type(msg) == ParamsMsg or type(msg) == GradientsMsg:
            # Params or gradients from child

            self.incoming_child_msgs.append(msg)

            # If all child updates are in, notify server thread
            if len(self.incoming_child_msgs) == len(self.children):
                with self.child_params_ready_cond:
                    self.child_params_ready = True
                    self.child_params_ready_cond.notify()

        elif type(msg) == ReplacementParamsMsg:
            # Params from parent

            self.incoming_parent_msgs.append(msg)

            # If all parent params are in, wake up server thread
            if len(self.incoming_parent_msgs) == len(self.parents):
                with self.parent_params_ready_cond:
                    self.parent_params_ready = True
                    self.parent_params_ready_cond.notify()


    def run(self):
        
        while True:

            # Wait until all children have sent in updates
            with self.child_params_ready_cond:
                while not self.child_params_ready:
                    self.child_params_ready_cond.wait()
                self.child_params_ready = False

            incoming_child_msgs_buffer = self.incoming_child_msgs
            self.incoming_child_msgs = []

            self.open_gantt()

            with self.params_lock:

                if self.update_policy == UpdatePolicy.AVERAGE:

                    # Average child param updates and assign to model cache
                    for param_id in self.params:
                        param_value = 0
                        for params_msg in incoming_child_msgs_buffer:
                            param_value += params_msg.params[param_id]
                        
                        param_value /= len(incoming_child_msgs_buffer)
                        self.params[param_id].assign(param_value)

                elif self.update_policy == UpdatePolicy.GRADIENT:

                    # Sum grads and apply
                    agg_grads = incoming_child_msgs_buffer[0]

                    for i in range(1, len(incoming_child_msgs_buffer)):
                        grads_msg = incoming_child_msgs_buffer[i]
                        for param_id in self.params:
                            agg_grads.gradients[param_id] += grads_msg.gradients[param_id]
                    
                    apply_list = []
                    for param_id in self.params:
                        apply_list.append((agg_grads.gradients[param_id], self.params[param_id]))

                    self.optimizer.apply_gradients(apply_list)
                    

            # Update parents
            params = self.get_params()

            self.close_gantt(GanttEvent.PARAM_UPDATE)

            for parent_id in self.parents:
                if self.parent_update_policies[parent_id] == UpdatePolicy.AVERAGE:
                    self.ni.send_params_average(self.id, parent_id, params)
                elif self.parent_update_policies[parent_id] == UpdatePolicy.GRADIENT:
                    self.ni.send_params_gradient(self.id, parent_id, agg_grads.gradients)

            # Get params from parents
            if len(self.parents) != 0:
                self.wait_for_parent_params()

            # Send to children
            for child_id in self.children:
                self.ni.ps_send_to_child(self.id, child_id, self.get_params())
