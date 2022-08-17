import os
from run_sim import load_config
import json


RUN_CMD = 'python run_sim.py'
TMP_PREPEND = 'tmp_'
EXPS_PATH = 'experiments/'


class ExpUtil:

    def __init__(self, config_file_name):
        self.base_config_file_name = config_file_name
        self.run_config_file_name = config_file_name
        self.config = load_config(os.path.join(EXPS_PATH, config_file_name))

    def __del__(self):
        self.remove_run_config_file()

    def remove_run_config_file(self):
        if self.run_config_file_name != self.base_config_file_name:
            os.remove(os.path.join(EXPS_PATH, self.run_config_file_name))

    def set_config(self, config_file_name):
        self.remove_run_config_file()

        self.base_config_file_name = config_file_name
        self.run_config_file_name = config_file_name
        self.config = load_config(os.path.join(EXPS_PATH, config_file_name))

    def export_config(self):
        self.remove_run_config_file()
        
        self.run_config_file_name = TMP_PREPEND + self.base_config_file_name
        run_config_file = open(os.path.join(EXPS_PATH, self.run_config_file_name), 'w')
        json.dump(self.config, run_config_file)

    def run(self):
        os.system(RUN_CMD + ' ' + os.path.join(EXPS_PATH, self.run_config_file_name))


def load_exp_util(config_file_name):
    return ExpUtil(config_file_name)