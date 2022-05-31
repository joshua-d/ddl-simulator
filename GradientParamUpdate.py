from ParamUpdate import ParamUpdate

class GradientParamUpdate(ParamUpdate):
    
    # gradients: [(grad, param_id)], from Worker.send_gradients
    def __init__(self, gradients, sender_id):
        super().__init__(True, sender_id)

        self.gradients = gradients

    def apply(self, params, optimizer):
        apply_list = []
        for grad, param_id in self.gradients:
            apply_list.append((grad, params[param_id]))

        optimizer.apply_gradients(apply_list)