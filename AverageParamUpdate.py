from ParamUpdate import ParamUpdate

class AverageParamUpdate(ParamUpdate):
    
    # new_params: { param_id: param value }
    def __init__(self, new_params, sender_id):
        super().__init__(True, sender_id)

        self.new_params = new_params


    def apply(self, params, optimizer):
        # Average new params into existing params
        for param_id in self.new_params:
            params[param_id].assign((params[param_id].value() + self.new_params[param_id]) / 2)