from trainless_stats import do_trainless
from NetworkSequenceGenerator import NetworkSequenceGenerator, ReceiveUpdateEvent
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

        do_trainless(config, 22_800_000, 300)
        print()
