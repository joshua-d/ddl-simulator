import json

from model_and_data_builder import model_builder, dataset_fn
from TwoPassCluster import TwoPassCluster

import sys


def load_config(config_file_path):
    with open(config_file_path) as config_file:
        config = json.load(config_file)
        config_file.close()
    return config


def run_sim(config):
    cluster = TwoPassCluster(model_builder, dataset_fn, config)
    cluster.start()



# TODO consider moving this to yet another file!
if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        config_file_path = 'config.json'

    config = load_config(config_file_path)
    run_sim(config)