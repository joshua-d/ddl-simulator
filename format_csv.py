import csv

infilename = "out.csv"
outfilename = "formatted.csv"
delimiter = '\t'


def load_csv(infilename):
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
        row_str += val + delimiter
    return row_str


def add_row(out_str, row_str):
    return out_str + row_str + '\n'



def examine_2d(csv_data, ver_key, hor_key, ex_key, isolates=None, ver_key_vals=None, hor_key_vals=None):
    pass


"""
isolates: { key_to_isolate: value_to_isolate_on }
first_ley_vals: [ val ]
    row's first key val must be in this list to be included
exact: bool
    if True, isolates must be properly configured such that there will be
    no 2 included rows with the same first key val
"""
def examine_1d(csv_data, hor_keys=None, isolates=None, first_key=None, first_key_vals=None, exact=True):
    if hor_keys is None:
        hor_keys = csv_data.keys()
    if first_key is None:
        first_key = hor_keys[0]
    if first_key_vals is None:
        first_key_vals = csv_data[first_key]
    if isolates is None:
        isolates = {}

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
                    if not found:
                        sorted_out_rows.append(row_idx)
                        found = True
                    else:
                        raise ValueError(f'Found row with duplicate first key, but exact set to True\nfirst key: {first_key}, row index: {row_idx}, value: {first_key_val}')
            if not found:
                raise ValueError('No row found with first key val "{first_key_val}", but exact set to True')
        out_rows = sorted_out_rows

    out_str = add_row('', make_row(hor_keys))

    for row_idx in out_rows:
        out_str = add_row(out_str, make_row([csv_data[key][row_idx] for key in hor_keys]))

    return out_str
    



def main():
    csv_data = load_csv(infilename)
    out_str = examine_1d(csv_data, ['topology', 'sync-config', 'bw', 'n-workers'], {'topology': '2-2-2-2'}, 'sync-config', ['S-S', 'S-A'])
    print(out_str)


if __name__ == '__main__':
    main()