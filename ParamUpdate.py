

class ParamUpdate:

    # return_params: bool, whether or not the updated node should send updated params back
    #   if false, sender_id not required
    def __init__(self, return_params, sender_id):
        self.return_params = return_params
        self.sender_id = sender_id

    def apply(self, params, optimizer):
        pass
    