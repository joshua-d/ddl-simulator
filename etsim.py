from TwoPassCluster import TwoPassCluster
from csv_to_configs import load_configs_csv, make_config, non_raw_config_keys
import datetime
from format_csv import make_row
from math import ceil
import sys
from importlib import import_module
from multiprocessing import Process


# ps_tsync_keys = [(f"ps-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'ps', config['nodes']))]
# w_tsync_keys = [(f"w-{node['id']}-tsync") for node in list(filter(lambda n: n['node_type'] == 'worker', config['nodes']))]
# keys = keys + ps_tsync_keys + w_tsync_keys


def run(config, stamp, out_keys, result_filename, output_dir):
    # Import model and data builder file
    madb = import_module(config['madb_file'])

    # Begin sim
    for i in range(config['n_runs']):
        cluster = TwoPassCluster(madb.model_builder, madb.train_dataset_fn, madb.test_dataset_fn, config, output_dir)
        new_stamp = stamp + '_' + str(i)
        
        # TODO model and stuff gets built event on trainless - inefficient, but doesn't take that much time
        # try:
        #     result_row = cluster.train(new_stamp) if not config['trainless'] else cluster.trainless(new_stamp)
        #     result_row_list = [result_row[key] for key in keys]
        # except Exception as e:
        #     print(e)
        #     result_row_list = []

        result_row = cluster.train(new_stamp) if not config['trainless'] else cluster.trainless(new_stamp)
        result_row_list = [result_row[key] for key in out_keys]


        with open(result_filename, 'a') as resfile:
            resfile.write(make_row(result_row_list) + '\n')
            resfile.close()

        print(f'> Completed run {i} out of {config["n_runs"]}')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'eval_logs'

    if len(sys.argv) > 1:
        configs_csv_filename = sys.argv[1]
    else:
        configs_csv_filename = 'configs.csv'

    configs = [make_config(raw_config) for raw_config in load_configs_csv(configs_csv_filename)]

    # Make time stamp
    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

    # Write key row to result file
    out_keys = list(configs[0]['raw_config'].keys())
    for key in non_raw_config_keys:
        if key not in out_keys:
            out_keys.append(key)

    result_filename = f"{output_dir}/results_{time_stamp}.csv"

    with open(result_filename, 'w') as resfile:
        resfile.write(make_row(out_keys) + '\n')
        resfile.close()

    sim_i = 0
    for config in configs:
        # run(config, time_stamp + '_' + str(sim_i), out_keys, result_filename, output_dir)
        p = Process(target=run, args=(config, time_stamp + '_' + str(sim_i), out_keys, result_filename, output_dir))
        p.start()
        p.join()
        del p
        # close() terminate()

        sim_i += 1
        print(time_stamp + f'\tCompleted sim {sim_i} out of {len(configs)}')