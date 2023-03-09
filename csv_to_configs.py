import csv, json

delimiter = '\t'


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


def make_config(global_config_json, raw_config):
    config = json.loads(global_config_json)

    config['raw_config'] = raw_config
    config['target_acc'] = float(raw_config['target-acc'])
    config['epochs'] = int(raw_config['epochs'])
    config['generate_gantt'] = bool(int(raw_config['generate-gantt']))
    config['trainless'] = bool(int(raw_config['trainless']))

    # Top level PS
    # TODO currently no support for different inbound/outbound bw
    config['nodes'].append({
        "node_type": "ps",
        "id": 0,
        "parent": None,
        "update_policy": "average",
        "sync_style": "sync" if raw_config['sync-config'][0] == 'S' else 'async',

        "aggr_time": 0,
        "apply_time": float(raw_config['ps-sync-time']) if raw_config['sync-config'][0] == 'S' else float(raw_config['ps-async-time']),

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
            "update_policy": "average",
            "sync_style": "sync" if raw_config['sync-config'][1] == 'S' else 'async',

            "aggr_time": 0,
            "apply_time": float(raw_config['ps-sync-time']) if raw_config['sync-config'][1] == 'S' else float(raw_config['ps-async-time']),

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

                "step_time": float(raw_config['w-step-time']),
                "st_variation": float(raw_config['w-step-var']),

                "inbound_bw": float(raw_config['bw']),
                "outbound_bw": float(raw_config['bw'])
            })
            node_id += 1
        parent_ps += 1

    return config
