

class ParameterServer:

    def __init__(self, params, optimizer):
        self.params = params
        self.optimizer = optimizer

    def on_request(self):
        vals_dict = {}
        for param_id in self.params:
            vals_dict[param_id] = self.params[param_id].value()
        return vals_dict

    def on_receive(self, gradients):
        params_list = []
        for param_id in gradients:
            params_list.append(self.params[param_id])
        grads_list = gradients.values()

        self.optimizer.apply_gradients(zip(grads_list, params_list))
