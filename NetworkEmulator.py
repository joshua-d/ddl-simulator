
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

        # (node_id -> inbound bw, node_id -> outbound bw)
        self.inbound_max, self.outbound_max = node_bws

        # { node_id -> [msgs] }
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
        self.timing_thread = Thread(target=self._process_sending_msgs, daemon=True)

        # Data transmission thread
        self.dt_thread = Thread(target=self._process_dtjq, daemon=True)


    def get_available_bw(self, msg_list, max_bw):

        if len(msg_list) == 0:
            return max_bw

        # Gather and sort send rates
        send_rates = []

        for msg in msg_list:
            send_rates.append(msg.send_rate)

        send_rates.sort(reverse=True)

        # Calc baseline bw
        baseline_bw = max_bw / len(send_rates)

        # Calc extra bandwidth and how much is in use
        extra_bw = 0
        extra_bw_in_use = 0

        for sr in send_rates:
            if sr < baseline_bw:
                extra_bw += baseline_bw - sr
            elif sr > baseline_bw:
                extra_bw_in_use += sr - baseline_bw

        # Start by giving free bw to avail
        avail_bw = extra_bw - extra_bw_in_use

        # Check base cases
        if avail_bw > send_rates[0]:
            return avail_bw

        # Equalize remaining extra bw
        sr_idx = 0
        n_splitting = 1

        while True:

            split_val = (send_rates[sr_idx] - avail_bw) * n_splitting / (n_splitting + 1)

            if sr_idx + 1 == len(send_rates) or avail_bw + split_val > send_rates[sr_idx + 1]:
                # Ready to split between current srs and avail
                avail_bw += split_val
                return avail_bw

            else:
                # Lower currs, add to avail, add sr_idx + 1 to split
                avail_bw += (send_rates[sr_idx] - send_rates[sr_idx + 1]) * n_splitting
                n_splitting += 1
                sr_idx += 1


    def add_msg(self, msg, msg_list, max_bw):

        if len(msg_list) == 0:
            msg_list.append(msg)
            return

        # Calc baseline bw
        baseline_bw = max_bw / len(msg_list)

        # Calc extra bandwidth and how much is in use
        extra_bw = 0
        extra_bw_in_use = 0

        for aux_msg in msg_list:
            sr = aux_msg.send_rate
            if sr < baseline_bw:
                extra_bw += baseline_bw - sr
            elif sr > baseline_bw:
                extra_bw_in_use += sr - baseline_bw

        # Start by giving free bw to pulled
        pulled_bw = extra_bw - extra_bw_in_use

        # Check base cases
        if pulled_bw > msg.send_rate:
            msg_list.append(msg)
            return

        # Pull remaining bw from top msgs
        msg_list.sort(key=lambda msg: msg.send_rate, reverse=True)

        msg_idx = 0
        n_pulling = 1

        while True:

            remaining = msg.send_rate - pulled_bw
            pull_result = msg_list[msg_idx].send_rate - (remaining / n_pulling)

            if msg_idx + 1 == len(msg_list) or pull_result > msg_list[msg_idx + 1].send_rate:
                # Ready to pull
                for j in range(msg_idx + 1):
                    msg_list[j].send_rate = pull_result
                msg_list.append(msg)
                return
            else:
                # Lower currs, add to pulled, add next to pulling
                pulled_bw += (msg_list[msg_idx].send_rate - msg_list[msg_idx + 1].send_rate) * n_pulling
                n_pulling += 1
                msg_idx += 1




    def send_msg(self, from_id, to_id, msg_size, dtj_fn):
        with self.sending_msgs_cond:

            from_avail = self.get_available_bw(self.outgoing[from_id], self.outbound_max[from_id])
            to_avail = self.get_available_bw(self.incoming[to_id], self.inbound_max[to_id])

            send_rate = min(from_avail, to_avail)
            msg = Message(from_id, to_id, msg_size, dtj_fn, 0, send_rate)

            # TODO thread safety here
            self.add_msg(msg, self.outgoing[from_id])
            self.add_msg(msg, self.incoming[to_id])

            msg.last_checked = time.perf_counter()
            self.sending_msgs.append(msg)
            self.sending_msgs_cond.notify()


    


    # Starting point of timing thread
    def _process_sending_msgs(self):
        while True:
            sent_msgs = []

            with self.sending_msgs_cond:
                while len(self.sending_msgs) == 0:
                    self.sending_msgs_cond.wait()

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
    def _process_dtjq(self):
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



class DM:
    def __init__(self, send_rate):
        self.send_rate = send_rate


if __name__ == '__main__':
    inbound = {
        0: 50,
        1: 50
    }
    outbound = {
        0: 60,
        1: 50
    }

    ne = NetworkEmulator((inbound, outbound))

    ne.outgoing[0].append(DM(5))
    ne.outgoing[0].append(DM(5))
    ne.outgoing[0].append(DM(15))
    ne.outgoing[0].append(DM(20))


    print(ne.get_available_bw(ne.outgoing[0], ne.outbound_max[0]))
    print()

    ne.add_msg(DM(10), ne.outgoing[0], ne.outbound_max[0])
    for msg in ne.outgoing[0]:
        print(msg.send_rate)