from threading import Lock, Thread, Condition
from Node import GanttEvent, Node, UpdatePolicy

from MessageTypes import *


class AsyncParameterServer(Node):

    def __init__(self, id, parents, update_policy, parent_update_policies, param_locations, ni, params, optimizer, update_interval, return_threshold):
        super().__init__(id, parents, parent_update_policies, param_locations, ni)

        # { param_id: tf.Variable }
        self.params = params 
        self.optimizer = optimizer

        self.update_interval = update_interval
        self.n_updates = 0

        self.return_threshold = return_threshold

        self.update_policy = update_policy

        # IDs of children - POPULATED IN CLUSTER
        self.children = []

        # used for get_test_model
        self.params_lock = Lock()

        self.server_thread = Thread(target=self.run, daemon=True)

        # Queue for param updates from children
        self.child_update_queue = []
        self.child_update_queue_cond = Condition()

        self.incoming_child_msgs = []
        self.incoming_child_msgs_cond = Condition()


    def handle_msg(self, msg):
        if type(msg) == ParamsMsg or type(msg) == GradientsMsg:
            # Params or gradients from child

            # Place in incoming_child_msgs and notify server thread
            with self.incoming_child_msgs_cond:
                self.incoming_child_msgs.append(msg)
                self.incoming_child_msgs_cond.notify()

        elif type(msg) == ReplacementParamsMsg:
            # Params from parent

            self.incoming_parent_msgs.append(msg)

            # If all parent params are in, wake up server thread
            if len(self.incoming_parent_msgs) == len(self.parents):
                with self.parent_params_ready_cond:
                    self.parent_params_ready = True
                    self.parent_params_ready_cond.notify()


    def wait_for_parent_params(self):
        # Wait until all params have come in
        with self.parent_params_ready_cond:
            while not self.parent_params_ready:
                self.parent_params_ready_cond.wait()
            self.parent_params_ready = False

        self.open_gantt()

        # Update own model cache
        for params_msg in self.incoming_parent_msgs:
            for param_id in params_msg.params:
                self.params[param_id].assign(params_msg.params[param_id])

        self.close_gantt(GanttEvent.PARAM_UPDATE)

        # Clear incoming_parent_msgs
        self.incoming_parent_msgs = []


    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def run(self):
        
        while True:

            # Wait for any messages from children
            with self.incoming_child_msgs_cond:
                while len(self.incoming_child_msgs) == 0:
                    self.incoming_child_msgs_cond.wait()
                incoming_child_msgs_buffer = self.incoming_child_msgs
                self.incoming_child_msgs = []


            with self.params_lock:

                if self.update_policy == UpdatePolicy.AVERAGE:

                    # If there is a parent, send updates straight up
                    # Parents may only have AVERAGE update policy
                    if len(self.parents) != 0:
                        for params_msg in incoming_child_msgs_buffer:

                            # Send updates to parents
                            for parent_id in self.parents:
                                parent_update_params = {}
                                for param_id in self.param_locations[parent_id]:
                                    parent_update_params[param_id] = params_msg.params[param_id]

                                self.ni.send_params_average(self.id, parent_id, parent_update_params)

                            # TODO don't need to store in model cache here - just relay straight down
                            # Get updates from parents
                            self.wait_for_parent_params()

                            # async, so params get sent to child after each params_msg
                            self.ni.ps_send_to_child(self.id, params_msg.from_id, self.get_params())

                    # Otherwise, average into model cache
                    else:
                        for params_msg in incoming_child_msgs_buffer:

                            self.open_gantt()

                            # Average into model cache
                            for param_id in self.params:
                                param_value = (self.params[param_id].value() + params_msg.params[param_id]) / 2
                                self.params[param_id].assign(param_value)

                            self.close_gantt(GanttEvent.PARAM_UPDATE)

                            # async, so params get sent to child after each params_msg
                            self.ni.ps_send_to_child(self.id, params_msg.from_id, self.get_params())


                elif self.update_policy == UpdatePolicy.GRADIENT:

                    # Parents may have AVERAGE or GRADIENT update policy!
                    if len(self.parents) != 0:
                        for grads_msg in incoming_child_msgs_buffer:

                            for parent_id in self.parents:
                                
                                if self.parent_update_policies[parent_id] == UpdatePolicy.AVERAGE:

                                    self.open_gantt()

                                    # optimize model cache with gradients
                                    apply_list = []
                                    for param_id in self.params:
                                        apply_list.append((grads_msg.gradients[param_id], self.params[param_id]))

                                    self.optimizer.apply_gradients(apply_list)

                                    # send params up
                                    parent_update_params = {}
                                    for param_id in self.param_locations[parent_id]:
                                        parent_update_params[param_id] = self.params[param_id]

                                    self.close_gantt(GanttEvent.PARAM_UPDATE)

                                    self.ni.send_params_average(self.id, parent_id, parent_update_params)

                                elif self.parent_update_policies[parent_id] == UpdatePolicy.GRADIENT:
                                    # relay gradients straight up
                                    parent_update_grads = {}
                                    for param_id in self.param_locations[parent_id]:
                                        parent_update_grads[param_id] = grads_msg.gradients[param_id]

                                    self.ni.send_params_gradient(self.id, parent_id, parent_update_grads)

                            # TODO see above todo
                            # Get updates from parents
                            self.wait_for_parent_params()

                            # async, so params get sent to child after each params_msg
                            self.ni.ps_send_to_child(self.id, grads_msg.from_id, self.get_params())

                    else:
                        for grads_msg in incoming_child_msgs_buffer:

                            self.open_gantt()

                            # optimize model cache with gradients
                            apply_list = []
                            for param_id in self.params:
                                apply_list.append((grads_msg.gradients[param_id], self.params[param_id]))

                            self.optimizer.apply_gradients(apply_list)

                            self.close_gantt(GanttEvent.PARAM_UPDATE)

                            # async, so params get sent to child after each params_msg
                            self.ni.ps_send_to_child(self.id, grads_msg.from_id, self.get_params())


    def start(self):
        self.msg_handler_thread.start()
        self.server_thread.start()
