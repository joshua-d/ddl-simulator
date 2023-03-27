import csv, json
from csv_to_configs import raw_config_keys, non_raw_config_keys

delimiter = '\t'
config_filename = "format_csv_config.json"


def load_csv(infilename, delimiter=delimiter):
    infile = open(infilename)

    csv_data = {}

    row_idx = 0
    for row in csv.reader(infile, delimiter=delimiter):
        if len(list(filter(lambda x: x != '', row))) != 0: # ignore space row
            if row_idx == 0:
                keys = row
                for key in row:
                    csv_data[key] = []
            else:
                key_idx = 0
                for val in row:
                    csv_data[keys[key_idx]].append(val)
                    key_idx += 1
            row_idx += 1

    return csv_data


def make_row(vals, delimiter=delimiter):
    row_str = ''
    for val in vals:
        row_str += str(val) + delimiter
    return row_str


def add_row(out_str, row_str):
    return out_str + row_str + '\n'


"""
ver_key_vals and hor_key_vals must be exact with no duplicates
"""
def examine_2d(csv_data, ver_key, hor_key, ex_key, isolates=None, ver_key_vals=None, hor_key_vals=None, **kwargs):
    if isolates is None:
        isolates = {}

    def check_isolates(row_idx):
        for iso_key in isolates:
            if csv_data[iso_key][row_idx] != isolates[iso_key]:
                return False
        return True

    out_rows = list(filter(check_isolates, [i for i in range(len(csv_data[ver_key]))]))

    # remove duplicates
    if ver_key_vals is None:
        ver_key_vals = []
        [ver_key_vals.append(csv_data[ver_key][row_idx]) for row_idx in out_rows if csv_data[ver_key][row_idx] not in ver_key_vals]
    if hor_key_vals is None:
        hor_key_vals = []
        [hor_key_vals.append(csv_data[hor_key][row_idx]) for row_idx in out_rows if csv_data[hor_key][row_idx] not in hor_key_vals]

    def check_key_vals(row_idx):
        if csv_data[ver_key][row_idx] not in ver_key_vals:
            return False
        if csv_data[hor_key][row_idx] not in hor_key_vals:
            return False
        return True
    
    out_rows = list(filter(check_key_vals, out_rows))

    out_str = add_row('', make_row([ex_key, hor_key]))
    out_str = add_row(out_str, make_row([ver_key] + hor_key_vals))

    for ver_key_val in ver_key_vals:
        ex_vals = []

        for hor_key_val in hor_key_vals:
            found = False
            for row_idx in out_rows:
                if csv_data[ver_key][row_idx] == ver_key_val and csv_data[hor_key][row_idx] == hor_key_val:
                    if found:
                        raise ValueError(f'Found row with duplicate vkey-hkey vals. Check isolates.\nvkey: {ver_key}, vval: {ver_key_val}, hkey: {hor_key}, hval: {hor_key_val}')
                    found = True
                    ex_vals.append(csv_data[ex_key][row_idx])

            # Leave blank if not found
            if not found:
            #     raise ValueError(f'No row found with proper vkey-hkey vals. Check isolates.\nvkey: {ver_key}, vval: {ver_key_val}, hkey: {hor_key}, hval: {hor_key_val}')
                ex_vals.append('')

        out_str = add_row(out_str, make_row([ver_key_val] + ex_vals))

    return out_str
                

"""
isolates: { key_to_isolate: value_to_isolate_on }
first_key_vals: [ val ]
    row's first key val must be in this list to be included
exact: bool
    if True, isolates must be properly configured such that there will be
    no 2 included rows with the same first key val, and first_key_vals must contain no duplicates
"""
def examine_1d(csv_data, hor_keys=None, isolates=None, first_key=None, first_key_vals=None, exact=False, sort_key=None, **kwargs):
    if hor_keys is None:
        hor_keys = csv_data.keys()
    if isolates is None:
        isolates = {}
    if first_key is None:
        first_key = hor_keys[0]
    if first_key_vals is None:
        first_key_vals = csv_data[first_key]
    

    hor_keys.remove(first_key)
    hor_keys = [first_key] + hor_keys

    def include_row(row_idx):
        if csv_data[first_key][row_idx] not in first_key_vals:
            return False
        for iso_key in isolates:
            if csv_data[iso_key][row_idx] != isolates[iso_key]:
                return False
        return True
    
    out_rows = list(filter(include_row, [i for i in range(len(csv_data[first_key]))]))

    if exact:
        sorted_out_rows = []
        for first_key_val in first_key_vals:
            found = False
            for row_idx in out_rows:
                if csv_data[first_key][row_idx] == first_key_val:
                    if found:
                        raise ValueError(f'Found row with duplicate first key, but exact set to True. Check isolates.\nfirst key: {first_key}, row index: {row_idx}, value: {first_key_val}')
                    sorted_out_rows.append(row_idx)
                    found = True
            if not found:
                raise ValueError('No row found with first key val "{first_key_val}", but exact set to True. Check isolates.')
        out_rows = sorted_out_rows

    out_str = add_row('', make_row(hor_keys))

    if sort_key is not None:
        out_rows.sort(key=lambda row_idx: csv_data[sort_key][row_idx])

    for row_idx in out_rows:
        out_str = add_row(out_str, make_row([csv_data[key][row_idx] for key in hor_keys]))

    return out_str


"""
Consolidates rows with same "consolidate keys" into one row by averaging avg_keys based on n-runs
Behavior of keys not in consolidate_keys or avg_keys and not the same val is undefined
"""
def consolidate(csv_data, consolidate_keys, avg_keys, **kwargs):
    new_data = {}
    for key in csv_data:
        new_data[key] = []

    consolidated_row_idxs = []

    for row_idx in range(len(csv_data[key])):
        if row_idx in consolidated_row_idxs:
            continue

        rows_to_consolidate = [row_idx]

        for aux_row_idx in range(len(csv_data[key])):
            if row_idx == aux_row_idx or aux_row_idx in consolidated_row_idxs:
                continue
            do_consolidate = True
            for cons_key in consolidate_keys:
                if csv_data[cons_key][row_idx] != csv_data[cons_key][aux_row_idx]:
                    do_consolidate = False
                    break
            
            if do_consolidate:
                rows_to_consolidate.append(aux_row_idx)

        cons_vals = {}
        total_n_runs = 0
        for cons_row_idx in rows_to_consolidate:
            total_n_runs += int(csv_data['n-runs'][cons_row_idx])
            for avg_key in avg_keys:
                if avg_key not in cons_vals:
                    cons_vals[avg_key] = 0
                cons_vals[avg_key] += float(csv_data[avg_key][cons_row_idx]) * int(csv_data['n-runs'][cons_row_idx])
            
            consolidated_row_idxs.append(cons_row_idx)

        for avg_key in cons_vals:
            cons_vals[avg_key] = round(cons_vals[avg_key] / total_n_runs, 4)

        for key in csv_data:
            if key in consolidate_keys:
                new_data[key].append(csv_data[key][row_idx])
            elif key in avg_keys:
                new_data[key].append(cons_vals[key])
            elif key == 'n-runs':
                new_data[key].append(total_n_runs)
            else:
                new_data[key].append(csv_data[key][row_idx])

    out_str = add_row('', make_row(new_data.keys()))

    for row_idx in range(len(new_data[key])):
        out_str = add_row(out_str, make_row([new_data[key][row_idx] for key in new_data]))

    return out_str


if __name__ == '__main__':
    fns = {
        'examine_2d': examine_2d,
        'examine_1d': examine_1d,
        'consolidate': consolidate
    }
    config = json.load(open(config_filename))

    csv_data = load_csv(config['infile'])

    if config['args']['consolidate_keys'] is None:
        raw_config_keys.remove('n-runs')
        consolidate_keys = raw_config_keys
    else:
        consolidate_keys = config['args']['consolidate_keys']

    if config['args']['avg_keys'] is None:
        non_raw_config_keys.remove('stamp')
        avg_keys = non_raw_config_keys
    else:
        avg_keys = config['args']['avg_keys']

    with open(config['outfile'], 'w') as outfile:
        outfile.write(fns[config['function']](
            csv_data,
            ver_key=config['args']['ver_key'],
            hor_key=config['args']['hor_key'],
            ex_key=config['args']['ex_key'],
            ver_key_vals=config['args']['ver_key_vals'],
            hor_key_vals=config['args']['hor_key_vals'],
            hor_keys=config['args']['hor_keys'],
            first_key=config['args']['first_key'],
            first_key_vals=config['args']['first_key_vals'],
            exact=config['args']['exact'],
            sort_key=config['args']['sort_key'],
            consolidate_keys=consolidate_keys,
            avg_keys=avg_keys,
            isolates=config['args']['isolates']
        ))


"""
Example config:
{
    "function": "examine_1d",
    "args": {
        "ver_key": "",
        "hor_key": "",
        "ex_key": "",
        "ver_key_vals": null,
        "hor_key_vals": null,

        "hor_keys": [],
        "first_key": null,
        "first_key_vals": null,
        "exact": null,
        "sort_key": null,

        "consolidate_keys": null,
        "avg_keys": [],

        "isolates": {

        }
    },
    "infile": "eval_logs/results_2023-03-27_15-59-15.csv",
    "outfile": "out.csv"
}
"""