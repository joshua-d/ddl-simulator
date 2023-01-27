from NetworkSequenceGenerator import NetworkSequenceGenerator, WorkerStepEvent, SendParamsEvent, ReceiveParamsEvent, PSAggrEvent, PSApplyEvent, PSParentApplyEvent
import datetime, json, sys


def make_worker_desc(n):
    return """
    {
            "node_type": "worker",
            "id": %d,
            "parent": 0,

            "step_time": 0.050,
            "st_variation": 0.050,

            "inbound_bw": 100,
            "outbound_bw": 100
    }
    """ % n


def load_config(config_file_path):
        with open(config_file_path) as config_file:
            config = json.load(config_file)
            config_file.close()
        return config



# Get effective bw from a node's individual eff
def parse_eff(eff):
    time_window = eff[-1][0] - eff[0][0]
    eff_bw = 0
    for i in range(len(eff) - 1):
        eff_bw += eff[i][1] * ((eff[i+1][0] - eff[i][0]) / time_window) # dsr * the fraction the time slice is of the entire window
    
    return eff_bw



if __name__ == '__main__':

    worker_nums = [1, 2, 4, 6, 8, 10, 12, 14, 16]

    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        config_file_path = 'config.json'

    config = load_config(config_file_path)

    for worker_num in worker_nums:

        # Remove all workers
        config['nodes'] = config['nodes'][0:1]

        # Fill workers
        for i in range(worker_num):
            config['nodes'].append(json.loads(make_worker_desc(i+1)))

        # Generate
        nsg = NetworkSequenceGenerator(config['nodes'], 22_800_000)

        for _ in range(5000):
            nsg.generate(0.001, 100.001)

        # Get tsync and BUC
        total_time = 0
        n_events = 0
        for event in nsg.events:
            if type(event) == ReceiveParamsEvent:
                total_time += event.end_time - event.start_time
                n_events += 1

        tsync = total_time / n_events
        buc = worker_num / tsync
        print(worker_num)
        print('tsync: %f' % tsync)
        print('BUC: %f' % buc)

        # Get PS eff bw
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
        print()

        # now = datetime.datetime.now()
        # time_str = str(now.time())
        # time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')
        # nsg.generate_gantt(time_stamp + str(worker_num))
