import threading


class ParameterServer:

    def __init__(self, params, optimizer):
        # TODO document this!

        # { param_id: tf.Variable }
        self.params = params
         
        self.optimizer = optimizer

        self.params_lock = threading.Lock()

    def on_request(self):
        vals_dict = {}
        for param_id in self.params:
            vals_dict[param_id] = self.params[param_id].value()
        return vals_dict

    def on_receive(self, gradients):
        apply_list = []
        for grad, param_id in gradients:
            apply_list.append((grad, self.params[param_id]))

        self.optimizer.apply_gradients(apply_list)
