import csv, json

delimiter = '\t'

raw_config_keys = [
    'topology',
    'sync_config',
    'bw',
    'w_step_time',
    'w_step_var',
    'ps_sync_time',
    'ps_async_time',
    'global_dropout_chance',

    'epochs',
    'target_acc',
    'stop_at_target',
    'eval_interval',
    'generate_gantt',
    'trainless',
    'bypass_NI',
    'n_runs',
    'node_config_file'

    'network_style',
    'update_type',
    'madb_file',
    'rb_strat'
]

non_raw_config_keys = [
    'n_workers',
    'n_mid_ps',

    'tpe',
    'final_acc',
    'e_to_target',
    't_to_target',
    'total_time',
    'avg_tsync',
    'wc_time',
    'stamp'
]

keys = raw_config_keys + non_raw_config_keys


def load_configs_csv(infilename, delimiter=delimiter):
    infile = open(infilename)

    raw_configs = []

    row_idx = 0
    for row in csv.reader(infile, delimiter=delimiter):
        if len(list(filter(lambda x: x != '', row))) != 0: # ignore space row
            if row_idx == 0:
                keys = row
            else:
                raw_configs.append({})
                key_idx = 0
                for val in row:
                    raw_configs[-1][keys[key_idx]] = val
                    key_idx += 1
                    
            row_idx += 1

    return raw_configs


def make_config(raw_config):
    config = {}

    config['raw_config'] = raw_config

    # Topology-independent sim controls
    config['epochs'] = int(raw_config['epochs'])
    config['target_acc'] = float(raw_config['target_acc'])
    config['stop_at_target'] = bool(int(raw_config['stop_at_target']))
    config['eval_interval'] = int(raw_config['eval_interval'])
    config['generate_gantt'] = bool(int(raw_config['generate_gantt']))
    config['trainless'] = bool(int(raw_config['trainless']))
    config['bypass_NI'] = bool(int(raw_config['bypass_NI']))
    config['n_runs'] = int(raw_config['n_runs'])
    
    
    config['update_type'] = raw_config['update_type']
    config['network_style'] = raw_config['network_style']
    config['madb_file'] = raw_config['madb_file']
    config['rb_strat'] = raw_config['rb_strat']    


    if 'node_config_file' in raw_config and raw_config['node_config_file'] not in ['none', '']:
        config['nodes'] = json.load(open(raw_config['node_config_file']))
    else:
        config['nodes'] = []

        # Top level PS
        # TODO currently no support for different inbound/outbound bw
        config['nodes'].append({
            "node_type": "ps",
            "id": 0,
            "parent": None,
            "sync_style": "sync" if raw_config['sync_config'][0] == 'S' else 'async',

            "aggr_time": 0,
            "apply_time": float(raw_config['ps_sync_time']) if raw_config['sync_config'][0] == 'S' else float(raw_config['ps_async_time']),

            "inbound_bw": float(raw_config['bw']),
            "outbound_bw": float(raw_config['bw'])
        })

        # Mid level PSs
        node_id = 1
        n_mid = raw_config['topology'].count('-') + 1 if raw_config['topology'].count('-') != 0 else 0
        for _ in range(n_mid):
            config['nodes'].append({
                "node_type": "ps",
                "id": node_id,
                "parent": 0,
                "sync_style": "sync" if raw_config['sync_config'][2] == 'S' else 'async',

                "aggr_time": 0,
                "apply_time": float(raw_config['ps_sync_time']) if raw_config['sync_config'][2] == 'S' else float(raw_config['ps_async_time']),

                "inbound_bw": float(raw_config['bw']),
                "outbound_bw": float(raw_config['bw'])
            })
            node_id += 1

        # Workers
        cluster_nums = []
        top = raw_config['topology']

        parent_ps = 0 # 1lvl

        while top.count('-') > 0: # 2lvl
            cluster_nums.append(int(top[0:top.find('-')]))
            top = top[top.find('-')+1:]
            parent_ps = 1

        cluster_nums.append(int(top))
        
        for n_cluster_workers in cluster_nums:
            for _ in range(n_cluster_workers):
                config['nodes'].append({
                    "node_type": "worker",
                    "id": node_id,
                    "parent": parent_ps,

                    "step_time": float(raw_config['w_step_time']),
                    "st_variation": float(raw_config['w_step_var']),
                    "dropout_chance": float(raw_config['global_dropout_chance']),

                    "inbound_bw": float(raw_config['bw']),
                    "outbound_bw": float(raw_config['bw'])
                })
                node_id += 1
            parent_ps += 1

    return config
