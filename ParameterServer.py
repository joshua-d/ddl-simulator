import threading


class ParameterServer:

    def __init__(self, id, params, optimizer, network):
        # TODO document this!

        self.id = id

        # { param_id: tf.Variable }
        self.params = params
         
        self.optimizer = optimizer

        self.network = network

        self.grads_queue = []
        self.waiting_workers = []
        self.grads_queue_cond = threading.Condition()

        self.stop_listening = False


    def get_params(self):
        # { param_id: param's value }
        vals_by_param_id = {}

        for param_id in self.params:
            vals_by_param_id[param_id] = self.params[param_id].value()
        
        return vals_by_param_id


    def apply_gradients(self, gradients):
        apply_list = []
        for grad, param_id in gradients:
            apply_list.append((grad, self.params[param_id]))

        self.optimizer.apply_gradients(apply_list)


    def start(self):
        self.stop_listening = False
        
        while not self.stop_listening:
            grads_queue_buffer, waiting_workers_buffer = self.network.wait_for_worker_request(self)

            # If there are any gradients, they must be applied before worker param requests are fulfilled - TODO this is a custom policy
            for grads in grads_queue_buffer:
                self.apply_gradients(grads)

            # Send params to any requesting workers
            if len(waiting_workers_buffer) > 0:
                vals_by_param_id = self.get_params() # TODO may need lock on here because cluster and self reading at same time?
                for wk_id in waiting_workers_buffer:
                    self.network.send_params(wk_id, vals_by_param_id)
