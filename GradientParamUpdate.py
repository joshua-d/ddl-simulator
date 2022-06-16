from ParamUpdate import ParamUpdate

class GradientParamUpdate(ParamUpdate):
    
    # gradients: Map of param ID to gradient
    def __init__(self, gradients, sender_id):
        super().__init__(True, sender_id)

        self.gradients = gradients

    def apply(self, params, optimizer):
        apply_list = []
        for param_id in self.gradients:
            apply_list.append((self.gradients[param_id], params[param_id]))

        optimizer.apply_gradients(apply_list)