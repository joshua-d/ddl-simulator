from model_and_data_builder import model_builder, dataset_fn
from TwoPassCluster import TwoPassCluster
from csv_to_configs import load_configs_csv, make_config, keys
import datetime
from format_csv import make_row
from math import ceil
import sys


global_config_json = """
{
    "bypass_NI": false,
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_train_samples": 50000,
    "num_test_samples": 50000,
	"network_style": "hd",
    "data_chunk_size": 64,
    "eval_interval": 782,
    "nodes": []
}
"""

# ps_tsync_keys = [(f"ps-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'ps', config['nodes']))]
# w_tsync_keys = [(f"w-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'worker', config['nodes']))]
# keys = keys + ps_tsync_keys + w_tsync_keys


def run(configs, stamp):
    # Write key row to result file
    result_filename = f"eval_logs/results_{stamp}.csv"

    with open(result_filename, 'w') as resfile:
        resfile.write(make_row(keys) + '\n')
        resfile.close()

    # Count total runs (for logging)
    total_runs = 0
    for config in configs:
        total_runs += config['n_runs']

    # Begin sims
    run_i = 0
    for config in configs:
        for _ in range(config['n_runs']):
            cluster = TwoPassCluster(model_builder, dataset_fn, config)
            new_stamp = stamp + '_' + str(run_i)
            # TODO model and stuff gets built event on trainless - inefficient, but doesn't take that much time
            try:
                result_row = cluster.train(new_stamp) if not config['trainless'] else cluster.trainless(new_stamp)
                result_row_list = [result_row[key] for key in keys]
            except Exception as e:
                print(e)
                result_row_list = []

            with open(result_filename, 'a') as resfile:
                resfile.write(make_row(result_row_list) + '\n')
                resfile.close()

            run_i += 1

            print(stamp + f'\tCompleted run {run_i} out of {total_runs}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        configs_csv_filename = sys.argv[1]
    else:
        configs_csv_filename = 'configs.csv'

    configs = [make_config(global_config_json, raw_config) for raw_config in load_configs_csv(configs_csv_filename)]

    # Make time stamp
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    run(configs, time_stamp)