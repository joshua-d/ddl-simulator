from ParameterServer import ParameterServer
import threading


# TODO Strictly assumes there is only 1 PS

class SyncParameterServer(ParameterServer):

    def __init__(self, id, params, optimizer, ni, num_workers):
        super().__init__(id, params, optimizer, ni)

        self.num_workers = num_workers
        self.round_num_grads_received = 0


    def reset_round(self):
        self.round_num_grads_received = 0
        

    def start(self):

        # Start by broadcasting params
        self.ni.broadcast_params(self.get_params())

        while True:

            # Wait for grads msg or params request
            grads_queue_buffer = self.ni.wait_for_grads(self)

            # Apply grads in order received
            for grads, wk_id in grads_queue_buffer:
                self.apply_gradients(grads)
                self.round_num_grads_received += 1

            # If all workers have sent in grads, send out params
            if self.round_num_grads_received == self.num_workers:
                self.round_num_grads_received = 0
                # print('broadcasting params')
                self.ni.broadcast_params(self.get_params())

