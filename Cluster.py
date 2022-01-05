import threading


class Cluster:

    def __init__(self):
        self.steps_completed = 0
        self.steps_scheduled = 0
        self.steps_completed_lock = threading.Lock()