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
        for key in self.timing:
            print(f'{key}: {self.timing[key]}')