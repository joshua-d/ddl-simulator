from Worker import Worker
import threading

# TODO strictly assumes there is only 1 PS

class SyncWorker(Worker):

    def send_gradients(self, gradients):
        for ps_id in self.cluster.param_locations:
            send_list = []
            for param_id in self.cluster.param_locations[ps_id]:
                send_list.append((gradients[param_id], param_id))
            
            ps = self.cluster.parameter_servers[ps_id]
            ps.on_receive(send_list)
        

    def train(self):
        print_lock = self.cluster.parameter_servers['ps0'].print_lock
        
        self.stop_training = False
        while not self.stop_training:

            # with print_lock:
            #     print('Worker %d completing step' % (self.id), flush=True)

            self.train_step()

            