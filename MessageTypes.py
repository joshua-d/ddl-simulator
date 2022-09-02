

class ParamsMsg:

    def __init__(self, params, from_id):
        self.params = params
        self.from_id = from_id


class GradientsMsg:

    def __init__(self, gradients, from_id):
        self.gradients = gradients
        self.from_id = from_id


class ReplacementParamsMsg:

    def __init__(self, params):
        self.params = params