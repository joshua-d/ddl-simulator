from NetworkSequenceGenerator import NetworkSequenceGenerator, UpdateType, WorkerStepEvent, ReceiveUpdateEvent
from time import perf_counter
from math import ceil

class TrainlessRunner:

    def __init__(self, config):

        self.config = config
        self._parse_config(config)

        self.num_workers = 0
        for node in self.node_descs:
            if node['node_type'] == 'worker':
                self.num_workers += 1

        self.nsg = NetworkSequenceGenerator(self.node_descs, self.model_size, self.network_style == 'hd', self.update_type, self.rb_strat, False)

    def _get_config_item(self, config, item):
        if item not in config:
            raise Exception('%s not in config' % item)
        else:
            return config[item]

    def _parse_config(self, config):

        self.network_style = self._get_config_item(config, 'network_style')

        self.node_descs = self._get_config_item(config, 'nodes')

        self.epochs = self._get_config_item(config, 'epochs')

        self.generate_gantt = self._get_config_item(config, 'generate_gantt')

        update_type = self._get_config_item(config, 'update_type')
        if update_type == 'grads':
            self.update_type = UpdateType.GRADS
        elif update_type == 'params':
            self.update_type = UpdateType.PARAMS
        else:
            raise Exception('invalid update_type in config')
        
        self.rb_strat = self._get_config_item(config, 'rb_strat')

        self.model_size = int(config['raw_config']['model_size'])
        self.num_train_samples = int(config['raw_config']['num_train_samples'])
        self.batch_size = int(config['raw_config']['batch_size'])


    def get_results(self, stamp, trainless, wc_time, end_time=None, avg_tsync=None, final_acc_test=None, ep_to_target_test=None, t_to_target_test=None, highest_acc_test=None, final_acc_train=None, ep_to_target_train=None, t_to_target_train=None, highest_acc_train=None, total_epochs=None):
        row = self.config['raw_config']

        row['n_runs'] = 1
        
        row['n_workers'] = self.num_workers
        row['n_mid_ps'] = len(list(filter(lambda node: node['node_type'] == 'ps', self.config['nodes']))) - 1

        if not trainless:
            row['final_acc_test'] = round(final_acc_test, 4)
            row['final_acc_train'] = round(final_acc_train, 4)
        else:
            row['final_acc_test'] = ''
            row['final_acc_train'] = ''

        if not trainless and ep_to_target_test is not None and t_to_target_test is not None:
            row['ep_to_target_test'] = round(ep_to_target_test, 4)
            row['t_to_target_test'] = round(t_to_target_test, 4)
        else:
            row['ep_to_target_test'] = ''
            row['t_to_target_test'] = ''

        if not trainless and ep_to_target_train is not None and t_to_target_train is not None:
            row['ep_to_target_train'] = round(ep_to_target_train, 4)
            row['t_to_target_train'] = round(t_to_target_train, 4)
        else:
            row['ep_to_target_train'] = ''
            row['t_to_target_train'] = ''

        if not trainless:
            row['highest_acc_test'] = round(highest_acc_test, 4)
            row['highest_acc_train'] = round(highest_acc_train, 4)
        else:
            row['highest_acc_test'] = ''
            row['highest_acc_train'] = ''


        # tpe
        if trainless:
            step_events = list(filter(lambda e: type(e) == WorkerStepEvent, self.nsg.events))
            end_time = 0
            for e in step_events:
                if e.end_time > end_time:
                    end_time = e.end_time

            row['tpe'] = round(end_time / self.epochs, 4)

        else:
            row['tpe'] = round(end_time / total_epochs, 4)
        
        
        row['total_sim_time'] = round(end_time, 4)


        # avg-tsync
        if trainless:
            receive_events = list(filter(lambda e: type(e) == ReceiveUpdateEvent, self.nsg.events))
            total_time = 0
            n_events = 0
            for event in receive_events:
                total_time += event.end_time - event.start_time
                n_events += 1

            avg_tsync = total_time / n_events
        
        row['avg_trans_time'] = round(avg_tsync, 4)

        row['wc_time'] = round(wc_time, 4)

        row['stamp'] = stamp

        return row


    def trainless(self, stamp):
        batches_per_epoch = self.num_train_samples / self.batch_size # TODO num train samples should be divisible by batch size

        start_wc_time = perf_counter()

        while not self.nsg.generate(end_batch=ceil(self.epochs * batches_per_epoch)):
            pass

        wc_time = perf_counter() - start_wc_time

        if self.generate_gantt:
            self.nsg.generate_gantt(stamp)

        return self.get_results(stamp, trainless=True, wc_time=wc_time)
