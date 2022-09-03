from enum import Enum
from threading import Condition, Thread, Lock
from time import perf_counter


class UpdatePolicy(Enum):
    REPLACE = 1
    GRADIENT = 2
    AVERAGE = 3


class GanttEvent(Enum):
    WORKER_STEP = 0
    PARAM_UPDATE = 1
    HANDLE_CHILD_UPDATE = 2


record_gantt = True


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
        self.gantt_buffer = []
        self.current_gantt = None
        self.gantt_start = None
        self.gantt_lock = Lock()


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


    # Waits for all params from parents to arrive, then updates own model cache
    def wait_for_parent_params(self):
        # Wait until all params have come in
        with self.parent_params_ready_cond:
            while not self.parent_params_ready:
                self.parent_params_ready_cond.wait()
            self.parent_params_ready = False

        self.open_gantt(GanttEvent.PARAM_UPDATE)

        # Update own model cache
        for params_msg in self.incoming_parent_msgs:
            for param_id in params_msg.params:
                self.params[param_id].assign(params_msg.params[param_id])

        self.close_gantt()

        # Clear incoming_parent_msgs
        self.incoming_parent_msgs = []


    def open_gantt(self, gantt_event):
        if record_gantt:
            self.current_gantt = gantt_event
            self.gantt_start = perf_counter()

    def close_gantt(self):
        if record_gantt:
            with self.gantt_lock:
                self.gantt_list.append((self.current_gantt, self.gantt_start, perf_counter()))

    def switch_gantt(self, gantt_event):
        if record_gantt:
            self.close_gantt()
            self.open_gantt(gantt_event)