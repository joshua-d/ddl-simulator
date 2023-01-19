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
            nsg.generate()

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

        # now = datetime.datetime.now()
        # time_str = str(now.time())
        # time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')
        # nsg.generate_gantt(time_stamp + str(worker_num))
