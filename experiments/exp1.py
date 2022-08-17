from experiments.util.exp_util import ExpUtil
from experiments.util.notify_pushbullet import notify_pushbullet

def main():
    exp_util = ExpUtil('config1.json')

    exp_util.run()
