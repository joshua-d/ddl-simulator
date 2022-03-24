import threading


class ParameterServer:

    def __init__(self, id, params, optimizer, nc):
        # TODO document this!

        self.id = id

        # { param_id: tf.Variable }
        self.params = params
         
        self.optimizer = optimizer

        self.nc = nc

        self.grads_queue = []
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

        # Start by broadcasting params
        self.nc.broadcast_params(self.get_params())
        
        while not self.stop_listening:
            grads_queue_buffer = self.nc.wait_for_grads(self)

            waiting_workers = []

            # Apply all grads before sending back - TODO this is a custom policy
            for grads, wk_id in grads_queue_buffer:
                self.apply_gradients(grads)
                waiting_workers.append(wk_id)

            # Send params to any requesting workers
            if len(waiting_workers) > 0:
                vals_by_param_id = self.get_params() # TODO may need lock on here because cluster and self reading at same time?
                for wk_id in waiting_workers:
                    self.nc.send_params(wk_id, vals_by_param_id)
                    
