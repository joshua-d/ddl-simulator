import csv

delimiter = '\t'


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
def examine_2d(csv_data, ver_key, hor_key, ex_key, isolates=None, ver_key_vals=None, hor_key_vals=None):
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
def examine_1d(csv_data, hor_keys=None, isolates=None, first_key=None, first_key_vals=None, exact=False, sort_key=None):
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
