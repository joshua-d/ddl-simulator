from model_and_data_builder import model_builder, dataset_fn
from TwoPassCluster import TwoPassCluster
from csv_to_configs import load_configs_csv, make_config
import datetime
from format_csv import make_row


configs_csv_filename = 'configs.csv'

keys = [
    'topology',
    'sync-config',
    'bw',
    'w-step-time',
    'w-step-var',
    'ps-sync-time',
    'ps-async-time',
    'epochs',
    'target-acc', # raw config ends here

    'n-workers',
    'n-mid-ps',

    'tpe',
    'e-to-target',
    't-to-target',
    'total-time',
    'avg-tsync'
]

# ps_tsync_keys = [(f"ps-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'ps', config['nodes']))]
# w_tsync_keys = [(f"w-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'worker', config['nodes']))]
# keys = keys + ps_tsync_keys + w_tsync_keys


if __name__ == '__main__':
    configs = [make_config(raw_config) for raw_config in load_configs_csv(configs_csv_filename)]

    # Make time stamp
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    # Write key row to result file
    result_filename = f"eval_logs/results_{time_stamp}.csv"

    with open(result_filename, 'w') as resfile:
        resfile.write(make_row(keys) + '\n')
        resfile.close()

    # Begin sims
    run_i = 0
    for config in configs:
        cluster = TwoPassCluster(model_builder, dataset_fn, config)
        result_row = cluster.start(time_stamp, run_i)

        result_row_list = [result_row[key] for key in keys]

        with open(result_filename, 'a') as resfile:
            resfile.write(make_row(result_row_list) + '\n')
            resfile.close()

        run_i += 1