from model_and_data_builder import model_builder, dataset_fn
from TwoPassCluster import TwoPassCluster
from csv_to_configs import load_configs_csv, make_config


configs_csv_filename = 'configs.csv'


if __name__ == '__main__':
    configs = [make_config(raw_config) for raw_config in load_configs_csv(configs_csv_filename)]

    for config in configs:
        cluster = TwoPassCluster(model_builder, dataset_fn, config)
        cluster.start()