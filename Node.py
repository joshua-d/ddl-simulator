from enum import Enum
from threading import Condition, Thread, Lock
from time import perf_counter

class GanttEvent(Enum):
    WORKING = 0
    PARAM_UPDATE = 1
    SENDING_PARAMS = 2
    RECEIVING_PARAMS = 3


class UpdatePolicy(Enum):
    REPLACE = 1
    GRADIENT = 2
    AVERAGE = 3


class Node:

    def __init__(self, id, parents, parent_update_policies, param_locations, ni):
        self.id = id

        # List of node IDs
        self.parents = parents

        # Map of parent node ID to update policy
        self.parent_update_policies = parent_update_policies

        # Map of parent node ID to param IDs that it holds
        self.param_locations = param_locations

        # Network Interface
        self.ni = ni

        self.msg_queue = []
        self.msg_queue_cond = Condition()

        self.msg_handler_thread = Thread(target=self.process_msgs, daemon=True)

        self.parent_params_ready = False
        self.parent_params_ready_cond = Condition()
        self.incoming_parent_msgs = []

        self.gantt_list = []
        self.gantt_start = None
        self.gantt_lock = Lock()
        self.record_gantt = ni.cluster.record_gantt


    def process_msgs(self):
        while True:
            with self.msg_queue_cond:
                while len(self.msg_queue) == 0:
                    self.msg_queue_cond.wait()

                msg_queue_buffer = self.msg_queue
                self.msg_queue = []

            for msg in msg_queue_buffer:
                self.handle_msg(msg)


    def handle_msg(self):
        pass
    
    def wait_for_parent_params(self):
        pass

    def open_gantt(self):
        if self.record_gantt:
            self.gantt_start = perf_counter()

    def close_gantt(self, gantt_event, data=None):
        if self.record_gantt:
            with self.gantt_lock:
                self.gantt_list.append((self.gantt_start, perf_counter(), gantt_event, data))
