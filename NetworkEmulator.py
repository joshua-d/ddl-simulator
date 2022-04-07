
from threading import Thread, Condition
import time


TIMING_THREAD_PERIOD = 0.010 # 10ms


class Message:
    def __init__(self, size, dtj_fn, last_checked):
        self.size = size
        self.dtj_fn = dtj_fn

        self.amt_sent = 0
        self.last_checked = last_checked


class NetworkEmulator:

    def __init__(self, bw):

        self.bw = bw

        # Data Transmission Job queue
        self.dtjq = []
        self.dtjq_cond = Condition()

        # Messages currently using bandwidth
        self.sending_msgs = []
        self.sending_msgs_cond = Condition()

        # Sending msg timing thread
        self.timing_thread = Thread(target=self.process_sending_msgs)
        self.timing_thread_waiting = True

        # Data transmission thread
        self.dt_thread = Thread(target=self.process_dtjq)
        self.dt_thread_waiting = True

        # Network idle utility. CURRENTLY SINGLE USE - once announce is set to True, once the network
        # becomes idle, network_idle will be set to true and the cond will notify.
        # network_idle will not be changed from True after this point
        self.network_idle = False
        self.network_idle_cond = Condition()
        self.announce_network_idle = False # if false, network_idle is not updated or announced
    

    def send_msg(self, msg_size, dtj_fn):
        with self.sending_msgs_cond:
            self.sending_msgs.append(Message(msg_size, dtj_fn, time.perf_counter()))
            self.sending_msgs_cond.notify()


    # Starting point of timing thread
    def process_sending_msgs(self):
        while True:
            sent_msgs = []

            with self.sending_msgs_cond:
                while len(self.sending_msgs) == 0:

                    self.timing_thread_waiting = True
                    if self.announce_network_idle and self.dt_thread_waiting:
                        with self.network_idle_cond:
                            self.network_idle = True
                            self.network_idle_cond.notify_all()

                    self.sending_msgs_cond.wait()

                self.timing_thread_waiting = False

                # print(len(self.sending_msgs))

                # TODO do I have to keep it locked here??? I think I do - r/w lock?
                current_time = time.perf_counter()
                avail_bw = self.bw / len(self.sending_msgs)

                msg_idx = 0
                while msg_idx < len(self.sending_msgs):
                    msg = self.sending_msgs[msg_idx]

                    msg.amt_sent += (current_time - msg.last_checked) * avail_bw
                    msg.last_checked = current_time

                    if msg.amt_sent >= msg.size:
                        # message has sent, remove from sending_msgs and add to sent_msgs
                        self.sending_msgs.pop(msg_idx)
                        msg_idx -= 1
                        sent_msgs.append(msg)

                    msg_idx += 1

            # Dispatch sent msgs to DT thread
            if len(sent_msgs) > 0:
                with self.dtjq_cond:
                    for sent_msg in sent_msgs:
                        self.dtjq.append(sent_msg.dtj_fn)
                    self.dtjq_cond.notify()


            time.sleep(TIMING_THREAD_PERIOD)


    # Starting point of DT thread
    def process_dtjq(self):
        while True:

            with self.dtjq_cond:
                while len(self.dtjq) == 0:

                    self.dt_thread_waiting = True
                    if self.announce_network_idle and self.timing_thread_waiting:
                        with self.network_idle_cond:
                            self.network_idle = True
                            self.network_idle_cond.notify_all()

                    self.dtjq_cond.wait()

                self.dt_thread_waiting = False

                # Move into buffer and unlock
                dtjq_buffer = self.dtjq
                self.dtjq = []

            for dtj_fn in dtjq_buffer:
                dtj_fn()


    def start(self):
        self.timing_thread.start()
        self.dt_thread.start()



    # For debugging use - sends all currently sending msgs right now
    # NOT TESTED
    def flush_messages(self):
        with self.sending_msgs_cond:
            msgs = self.sending_msgs
            self.sending_msgs = []

        with self.dtjq_cond:
            for dtj_fn in self.dtjq:
                dtj_fn()
            for msg in msgs:
                msg.dtj_fn()
