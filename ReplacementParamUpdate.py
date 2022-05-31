from ParamUpdate import ParamUpdate

class ReplacementParamUpdate(ParamUpdate):
    
    # replacer_params: { param_id: param value }
    def __init__(self, replacer_params):
        super().__init__(False, None)

        self.replacer_params = replacer_params


    def apply(self, params, optimizer):
        for param_id in self.replacer_params:
            params[param_id].assign(self.replacer_params[param_id])