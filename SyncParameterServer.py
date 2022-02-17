from ParameterServer import ParameterServer
import threading


class SyncParameterServer(ParameterServer):

    def __init__(self, id, params, optimizer, network, num_workers):
        super().__init__(id, params, optimizer, network)

        self.num_workers = num_workers
        self.round_num_grads_received = 0


    def reset_round(self):
        self.round_num_grads_received = 0
        self.network.flush_worker_params_queues() # TODO maybe add logic in send params so that we don't need this - send params only if params queue is empty
        
        # Send out initial params
        vals_by_param_id = self.get_params()
        for wk_id in range(self.num_workers): # ***ASSUMES WORKER IDS ARE INDEX IN CLUSTER.WORKERS***
            self.network.send_params(wk_id, vals_by_param_id)


    def start(self):
        self.stop_listening = False

        self.reset_round()
        
        while not self.stop_listening:

            # Wait for grads msg or params request
            grads_queue_buffer, waiting_workers_buffer = self.network.wait_for_worker_request(self)

            # Apply grads in order received
            for grads in grads_queue_buffer:
                self.apply_gradients(grads)
                self.round_num_grads_received += 1

            # Send params to any requesting workers - THIS IS ONLY USED FOR GETTING TEST MODEL
            if len(waiting_workers_buffer) > 0:
                vals_by_param_id = self.get_params()
                for wk_id in waiting_workers_buffer:
                    self.network.send_params(wk_id, vals_by_param_id)

            # If all workers have sent in grads, send out params
            if self.round_num_grads_received == self.num_workers:
                self.round_num_grads_received = 0
                vals_by_param_id = self.get_params()
                for wk_id in range(self.num_workers): # ***ASSUMES WORKER IDS ARE INDEX IN CLUSTER.WORKERS***
                    self.network.send_params(wk_id, vals_by_param_id)

