
from threading import Thread, Condition
import time
from math import inf


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

        self.total_msgs = 0  # TODO could perhaps use len(self.sending_msgs)

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


    def _update_send_rates(self):

        # Prepare
        incoming_offering = {}
        outgoing_offering = {}

        for node_id in self.incoming.keys():
            for msg in self.incoming[node_id]:
                incoming_offering[msg] = self.inbound_max[node_id] / len(self.incoming[node_id])
            for msg in self.outgoing[node_id]:
                outgoing_offering[msg] = self.outbound_max[node_id] / len(self.outgoing[node_id])

        final_msgs = []  # TODO perhaps more efficient if this was a map?

        # Begin updating
        while len(final_msgs) != self.total_msgs:

            # Find least
            least_offering = inf
            for msg in incoming_offering.keys():
                if msg not in final_msgs:
                    if incoming_offering[msg] < least_offering:
                        least_offering = incoming_offering[msg]
                        least_offering_node_id = msg.to_id
                        incoming = True
                    if outgoing_offering[msg] < least_offering:
                        least_offering = outgoing_offering[msg]
                        least_offering_node_id = msg.from_id
                        incoming = False

            # Mark node's msgs as final, follow, distribute
            if incoming:
                for msg in self.incoming[least_offering_node_id]:
                    if msg not in final_msgs:
                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.outgoing[msg.from_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (outgoing_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                outgoing_offering[aux_msg] += distribute_amt
                    
            else:
                for msg in self.outgoing[least_offering_node_id]:
                    if msg not in final_msgs:
                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.incoming[msg.to_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (incoming_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                incoming_offering[aux_msg] += distribute_amt



    def send_msg(self, from_id, to_id, msg_size, dtj_fn):
        with self.sending_msgs_cond:
            msg = Message(from_id, to_id, msg_size, dtj_fn, time.perf_counter(), 0)
            
            self.outgoing[from_id].append(msg)
            self.incoming[to_id].append(msg)
            self.total_msgs += 1

            self._update_send_rates()

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
                        self.total_msgs -= 1
                        self._update_send_rates()

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

