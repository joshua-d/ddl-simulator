from trainless_stats import do_trainless
from NetworkSequenceGenerator import NetworkSequenceGenerator, ReceiveParamsEvent
import datetime, json, sys
from math import floor


def load_config(config_file_path):
        with open(config_file_path) as config_file:
            config = json.load(config_file)
            config_file.close()
        return config



def do44():
    # S-S
    config['nodes'][0]['sync_style'] = 'sync'
    config['nodes'][1]['sync_style'] = 'sync'
    config['nodes'][2]['sync_style'] = 'sync'

    print('S-S')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # A-S
    config['nodes'][0]['sync_style'] = 'async'
    config['nodes'][1]['sync_style'] = 'sync'
    config['nodes'][2]['sync_style'] = 'sync'

    print('A-S')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # S-A
    config['nodes'][0]['sync_style'] = 'sync'
    config['nodes'][1]['sync_style'] = 'async'
    config['nodes'][2]['sync_style'] = 'async'

    print('S-A')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # A-A
    config['nodes'][0]['sync_style'] = 'async'
    config['nodes'][1]['sync_style'] = 'async'
    config['nodes'][2]['sync_style'] = 'async'

    print('A-A')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()


def do2222():
    # S-S
    config['nodes'][0]['sync_style'] = 'sync'
    config['nodes'][1]['sync_style'] = 'sync'
    config['nodes'][2]['sync_style'] = 'sync'
    config['nodes'][3]['sync_style'] = 'sync'
    config['nodes'][4]['sync_style'] = 'sync'

    print('S-S')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # A-S
    config['nodes'][0]['sync_style'] = 'async'
    config['nodes'][1]['sync_style'] = 'sync'
    config['nodes'][2]['sync_style'] = 'sync'
    config['nodes'][3]['sync_style'] = 'sync'
    config['nodes'][4]['sync_style'] = 'sync'

    print('A-S')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # S-A
    config['nodes'][0]['sync_style'] = 'sync'
    config['nodes'][1]['sync_style'] = 'async'
    config['nodes'][2]['sync_style'] = 'async'
    config['nodes'][3]['sync_style'] = 'async'
    config['nodes'][4]['sync_style'] = 'async'

    print('S-A')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()

    # A-A
    config['nodes'][0]['sync_style'] = 'async'
    config['nodes'][1]['sync_style'] = 'async'
    config['nodes'][2]['sync_style'] = 'async'
    config['nodes'][3]['sync_style'] = 'async'
    config['nodes'][4]['sync_style'] = 'async'

    print('A-A')
    print()
    do_trainless(config, 22_800_000, None, 520)
    print()


if __name__ == '__main__':

    # 4-4
    config = load_config('config44.json')

    # low bw

    # low work

    for node in config['nodes']:
        node['inbound_bw'] = 100
        node['outbound_bw'] = 100

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 1
            node['st_variation'] = 0.250

    print('low bw low work')
    print()
    do44()
    print()

    # high work

    for node in config['nodes']:
        node['inbound_bw'] = 100
        node['outbound_bw'] = 100

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 2
            node['st_variation'] = 0.500

    print('low bw high work')
    print()
    do44()
    print()

    # high bw

    # low work

    for node in config['nodes']:
        node['inbound_bw'] = 1000
        node['outbound_bw'] = 1000

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 1
            node['st_variation'] = 0.250

    print('high bw low work')
    print()
    do44()
    print()

    # high work

    for node in config['nodes']:
        node['inbound_bw'] = 1000
        node['outbound_bw'] = 1000

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 2
            node['st_variation'] = 0.500

    print('high bw high work')
    print()
    do44()
    print()



    # 2-2-2-2
    config = load_config('config2222.json')

    # low bw

    # low work

    for node in config['nodes']:
        node['inbound_bw'] = 100
        node['outbound_bw'] = 100

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 1
            node['st_variation'] = 0.250

    print('low bw low work')
    print()
    do2222()
    print()

    # high work

    for node in config['nodes']:
        node['inbound_bw'] = 100
        node['outbound_bw'] = 100

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 2
            node['st_variation'] = 0.500

    print('low bw high work')
    print()
    do2222()
    print()

    # high bw

    # low work

    for node in config['nodes']:
        node['inbound_bw'] = 1000
        node['outbound_bw'] = 1000

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 1
            node['st_variation'] = 0.250

    print('high bw low work')
    print()
    do2222()
    print()

    # high work

    for node in config['nodes']:
        node['inbound_bw'] = 1000
        node['outbound_bw'] = 1000

        if node['node_type'] == 'ps':
            node['aggr_time'] = 0
            node['apply_time'] = 0.010

        if node['node_type'] == 'worker':
            node['step_time'] = 2
            node['st_variation'] = 0.500

    print('high bw high work')
    print()
    do2222()
    print()
