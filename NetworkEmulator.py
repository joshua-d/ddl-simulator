
from threading import Thread, Condition
import time


TIMING_THREAD_PERIOD = 0.001 # 1ms


class Message:
    def __init__(self, from_id, to_id, size, dtj_fn, last_checked, send_rate):
        self.from_id = from_id
        self.to_id = to_id
        self.size = size
        self.dtj_fn = dtj_fn

        self.amt_sent = 0
        self.last_checked = last_checked

        self.send_rate = send_rate


class NetworkEmulator:

    def __init__(self, node_bws):

        self.inbound_max, self.outbound_max = node_bws

        self.incoming = {}
        self.outgoing = {}

        for node_id in self.inbound_max.keys():
            self.incoming[node_id] = []
            self.outgoing[node_id] = []

        # Data Transmission Job queue
        self.dtjq = []
        self.dtjq_cond = Condition()

        # Messages currently using bandwidth
        self.sending_msgs = []
        self.sending_msgs_cond = Condition()

        # Sending msg timing thread
        self.timing_thread = Thread(target=self.process_sending_msgs, daemon=True)

        # Data transmission thread
        self.dt_thread = Thread(target=self.process_dtjq, daemon=True)


    def _update_send_rate(self, msg):
        outbound_bw = self.outbound_max[msg.from_id] / len(self.outgoing[msg.from_id])
        inbound_bw = self.inbound_max[msg.to_id] / len(self.incoming[msg.to_id])

        msg.send_rate = min(outbound_bw, inbound_bw)


    def send_msg(self, from_id, to_id, msg_size, dtj_fn):
        with self.sending_msgs_cond:
            msg = Message(from_id, to_id, msg_size, dtj_fn, time.perf_counter(), 0)
            
            self.outgoing[from_id].append(msg)
            self.incoming[to_id].append(msg)

            for aux_msg in self.outgoing[from_id]:
                self._update_send_rate(aux_msg)

            for aux_msg in self.incoming[to_id]:
                self._update_send_rate(aux_msg)

            self.sending_msgs.append(msg)
            self.sending_msgs_cond.notify()


    # Starting point of timing thread
    def process_sending_msgs(self):
        while True:
            sent_msgs = []

            with self.sending_msgs_cond:
                while len(self.sending_msgs) == 0:
                    self.sending_msgs_cond.wait()

                # print(len(self.sending_msgs))

                # TODO do I have to keep it locked here??? I think I do - r/w lock?
                current_time = time.perf_counter()

                msg_idx = 0
                while msg_idx < len(self.sending_msgs):
                    msg = self.sending_msgs[msg_idx]

                    msg.amt_sent += (current_time - msg.last_checked) * msg.send_rate
                    msg.last_checked = current_time

                    if msg.amt_sent >= msg.size:
                        # message has sent, remove from sending_msgs and add to sent_msgs
                        self.sending_msgs.pop(msg_idx)
                        msg_idx -= 1
                        sent_msgs.append(msg)

                        # remove from incoming/outgoing and update send rates
                        self.outgoing[msg.from_id].remove(msg)
                        self.incoming[msg.to_id].remove(msg)

                        for aux_msg in self.outgoing[msg.from_id]:
                            self._update_send_rate(aux_msg)

                        for aux_msg in self.incoming[msg.to_id]:
                            self._update_send_rate(aux_msg)

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
                    self.dtjq_cond.wait()

                # Move into buffer and unlock
                dtjq_buffer = self.dtjq
                self.dtjq = []

            for dtj_fn in dtjq_buffer:
                dtj_fn()


    def start(self):
        self.timing_thread.start()
        self.dt_thread.start()


    # def wait_for_idle(self):
    #     with self.sending_msgs_cond:
    #         while len(self.sending_msgs) > 0:
    #             self.sending_msgs_cond.wait()

    #     with self.dtjq_cond:
    #         while len(self.dtjq) > 0:
    #             self.dtjq_cond.wait()



    # def pause_and_clear(self):
    #     self.paused = True
    #     with self.dtjq_cond:
    #         self.dtjq = []
    #     with self.sending_msgs_cond:
    #         self.sending_msgs = []


    # def resume(self):
    #     self.paused = False



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
