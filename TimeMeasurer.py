from time import perf_counter

class TimeMeasurer:
    
    def __init__(self, keys):
        self.timing = {}
        for key in keys:
            self.timing[key] = 0

        self.curr_key = None
        self.start_time = None

    def start(self, key):
        if self.curr_key is not None: # TODO REMOVE THIS
            raise ValueError('double timing start')
        self.curr_key = key
        self.start_time = perf_counter()

    def end(self):
        self.timing[self.curr_key] += perf_counter() - self.start_time
        self.curr_key = None # TODO AND THIS

    def get(self):
        return self.timing
    
    def print(self):
        total = 0

        for key in self.timing:
            print(f'{key}\t{self.timing[key]}')
            total += self.timing[key]

        print()

        train_percent = round((self.timing['fp'] + self.timing['opt']) / total * 100, 2)
        param_percent = round((self.timing['param_assign'] + self.timing['param_aggr']) / total * 100, 2)
        batch_percent = round((self.timing['batch_fetch']) / total * 100, 2)
        eval_percent = round((self.timing['eval'] + self.timing['logging']) / total * 100, 2)
        nsg_percent = round((self.timing['nsg_gen']) / total * 100, 2)
        other_percent = round((self.timing['other']) / total * 100, 2)

        print(f'Training\t{train_percent}%')
        print(f'Param aggregation\t{param_percent}%')
        print(f'Batch fetching\t{batch_percent}%')
        print(f'Eval/logging\t{eval_percent}%')
        print(f'First pass generation\t{nsg_percent}%')
        print(f'Other overhead\t{other_percent}%')