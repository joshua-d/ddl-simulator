from NetworkSequenceGenerator import NetworkSequenceGenerator, WorkerStepEvent, ReceiveUpdateEvent
import datetime, json, sys


def load_config(config_file_path):
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()
    return config


def modified_generate(self, end_time, end_batch=None, eff_start=None, eff_end=None):
    # Move NE until a msg has sent
    sent_msgs = self.ne.move(eff_start, eff_end)

    while len(sent_msgs) == 0:
        if (end_time is not None and self.ne.current_time >= end_time) or (end_batch is not None and self.n_batches >= end_batch):
            return True
        sent_msgs = self.ne.move(eff_start, eff_end)
        
    # Process sent msgs
    for msg in sent_msgs:
        self._process_msg(msg)

    if (end_time is not None and self.ne.current_time >= end_time) or (end_batch is not None and self.n_batches >= end_batch):
        return True

    return False


# Get effective bw from a node's individual eff
def parse_eff(eff):
    time_window = eff[-1][0] - eff[0][0]
    eff_bw = 0
    for i in range(len(eff) - 1):
        eff_bw += eff[i][1] * ((eff[i+1][0] - eff[i][0]) / time_window) # dsr * the fraction the time slice is of the entire window
    
    return eff_bw


def do_trainless(config, model_size, end_time, end_batch=None, eff_start=None, eff_end=None):
    worker_num = len(list(filter(lambda n: n['node_type'] == 'worker', config['nodes'])))
    print(f'{worker_num} workers, {end_time} seconds')

    # Generate
    nsg = NetworkSequenceGenerator(config['nodes'], model_size, config['network_style'] == 'hd')

    while not modified_generate(nsg, end_time, end_batch, eff_start, eff_end):
        pass

    # Get n batches and end time
    step_events = list(filter(lambda e: type(e) == WorkerStepEvent, nsg.events))
    step_events.sort(key=lambda e: e.start_time)

    if end_batch is not None:
        step_events = step_events[0:end_batch]
    
    n_batches = len(step_events)

    if end_time is None:
        end_time = 0
        for e in step_events:
            if e.end_time > end_time:
                end_time = e.end_time

    bps = n_batches / end_time
    print(f'n batches: {n_batches}')
    print(f'end time: {end_time}')
    print(f'bps: {bps}')

    # Get tsync and BUC
    receive_events = list(filter(lambda e: type(e) == ReceiveUpdateEvent, nsg.events))
    total_time = 0
    n_events = 0
    for event in receive_events:
        total_time += event.end_time - event.start_time
        n_events += 1

    tsync = total_time / n_events
    buc = worker_num / tsync
    print('tsync: %f' % tsync)
    print('BUC: %f' % buc)

    # Get PS eff bw
    if eff_start is not None:
        ps_eff_in = parse_eff(nsg.ne.eff_in[0])
        ps_eff_out = parse_eff(nsg.ne.eff_out[0])
        
        avg_w_eff_in = 0
        avg_w_eff_out = 0

        for worker_id in range(1, worker_num + 1):
            w_eff_in = parse_eff(nsg.ne.eff_in[worker_id])
            w_eff_out = parse_eff(nsg.ne.eff_out[worker_id])
            avg_w_eff_in += w_eff_in
            avg_w_eff_out += w_eff_out

        avg_w_eff_in /= worker_num
        avg_w_eff_out /= worker_num

        print('(Mbps)')
        print('PS eff bw in: %f' % (ps_eff_in / 1_000_000))
        print('PS eff bw out: %f' % (ps_eff_out / 1_000_000))

        print('Avg worker eff bw in: %f' % (avg_w_eff_in / 1_000_000))
        print('Avg worker eff bw out: %f' % (avg_w_eff_out / 1_000_000))
        print('sum: %f' % ((avg_w_eff_in + avg_w_eff_out) / 1_000_000))

    if config['generate_gantt']:
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')
        nsg.generate_gantt(time_stamp + '_' + str(worker_num))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        config_file_path = 'config.json'

    config = load_config(config_file_path)
    do_trainless(config, 22_800_000, 60)

    

        
