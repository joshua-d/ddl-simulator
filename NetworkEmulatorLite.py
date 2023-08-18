from math import inf, isclose, sqrt

# Linear growth coefficient, factor multiplied with dsr
base_lgc = 1

# Starting send rate, b/s
starting_sr = 1

# How often to increment sr, s
sr_update_period = 0.002

class Message:
    def __init__(self, from_id, to_id, size, in_time):
        self.from_id = from_id
        self.to_id = to_id
        self.size = size

        self.amt_sent = 0
        self.in_time = in_time
        self.last_checked = in_time
        self.last_sr_update = in_time

        # Designated send rate
        self.dsg_send_rate = 0

        # Current lgc
        self.lgr = 0

        # Current send rate
        self.send_rate = 0

        self.start_time = 0
        self.end_time = 0


class NetworkEmulatorLite:

    def __init__(self, node_bws, half_duplex):

        self.inbound_max, self.outbound_max = node_bws
        self.half_duplex = half_duplex

        self.sending = {}
        self.receiving = {}

        self.nodes = self.inbound_max.keys()

        for node_id in self.nodes:
            self.sending[node_id] = []
            self.receiving[node_id] = []

        self.total_msgs = 0  # TODO could perhaps use len(self.sending_msgs)

        # Messages currently using bandwidth
        self.sending_msgs = []

        self.current_time = 0

        self.future_msgs = []

        # Effective bandwidth measurement
        self.eff_in = {}
        self.eff_out = {}

        for node_id in self.nodes:
            self.eff_in[node_id] = []
            self.eff_out[node_id] = []


    def _update_dsg_send_rates(self):

        # Prepare
        incoming_offering = {}
        outgoing_offering = {}

        for node_id in self.nodes:
            for msg in self.receiving[node_id]:
                
                # This part simulates half duplex
                if self.half_duplex and len(self.sending[node_id]) != 0:
                    inbound_max = self.inbound_max[node_id] / 2
                else:
                    inbound_max = self.inbound_max[node_id]

                incoming_offering[msg] = inbound_max / len(self.receiving[node_id])

            for msg in self.sending[node_id]:

                # This part simulates half duplex
                if self.half_duplex and len(self.receiving[node_id]) != 0:
                    outbound_max = self.outbound_max[node_id] / 2
                else:
                    outbound_max = self.outbound_max[node_id]

                outgoing_offering[msg] = outbound_max / len(self.sending[node_id])

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
                for msg in self.receiving[least_offering_node_id]:
                    if msg not in final_msgs:

                        msg.dsg_send_rate = least_offering
                        msg.lgr = base_lgc * msg.dsg_send_rate
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.sending[msg.from_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (outgoing_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                outgoing_offering[aux_msg] += distribute_amt
                    
            else:
                for msg in self.sending[least_offering_node_id]:
                    if msg not in final_msgs:

                        msg.dsg_send_rate = least_offering
                        msg.lgr = base_lgc * msg.dsg_send_rate
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.receiving[msg.to_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (incoming_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                incoming_offering[aux_msg] += distribute_amt


    def send_msg(self, from_id, to_id, msg_size, in_time):
        msg = Message(from_id, to_id, msg_size, in_time)

        self.future_msgs.append(msg)


    # msg must be sending
    def _predict_amt_sent(self, msg, time):
        t = time - self.current_time
        return msg.amt_sent + msg.send_rate * t + msg.lgr * t * t
    

    def _get_completion_time(self, msg):
        a = msg.lgr
        b = msg.send_rate
        c = -(msg.size - msg.amt_sent)

        discriminant = b**2 - 4*a*c
        x1 = (-b + sqrt(discriminant)) / (2*a)
        x2 = (-b - sqrt(discriminant)) / (2*a)
        return max(x1, x2) + self.current_time


    # TODO make sure using inf does not take a lot of time
    # TODO: eff_start cannot be 0
    def move(self, timing, eff_start=None, eff_end=None):

        timing.start('find_earliest')

        # Find next earliest in time
        earliest_in_time = inf

        for msg in self.future_msgs:
            if msg.in_time < earliest_in_time:
                earliest_in_time = msg.in_time

        # Get amt sent at next earliest in time for each msg
        for msg in self.sending_msgs:
            pass

        # Find next earliest completion time
        earliest_completion_time = inf

        for msg in self.sending_msgs:
            msg_completion_time = self.current_time + (msg.size - msg.amt_sent) / msg.send_rate

            if msg_completion_time < earliest_completion_time:
                earliest_completion_time = msg_completion_time

        # Find next earliest sr update time
        earliest_sr_update_time = inf

        for msg in self.sending_msgs:
            msg_sr_update_time = msg.last_sr_update + sr_update_period
            if msg_sr_update_time < earliest_sr_update_time:
                earliest_sr_update_time = msg_sr_update_time

        

        timing.end()

        # Move to next earliest completion time or in time (or eff checkpoint!)
        if eff_start is not None:
            if self.current_time < eff_start:
                self.current_time = min(earliest_completion_time, earliest_in_time, earliest_sr_update_time, eff_start)
            elif self.current_time < eff_end:
                self.current_time = min(earliest_completion_time, earliest_in_time, earliest_sr_update_time, eff_end)
            else:
                self.current_time = min(earliest_completion_time, earliest_in_time, earliest_sr_update_time)
        else:
            self.current_time = min(earliest_completion_time, earliest_in_time, earliest_sr_update_time)

        timing.start('process_sending')
        # Process sending msgs
        sent_msgs = []

        msg_idx = 0
        while msg_idx < len(self.sending_msgs):
            msg = self.sending_msgs[msg_idx]

            # Update msg amt_sent, last_checked, and [current] send_rate
            msg.amt_sent += (self.current_time - msg.last_checked) * msg.send_rate
            msg.last_checked = self.current_time

            if msg.amt_sent > msg.size or isclose(msg.amt_sent, msg.size):
                # message has sent, remove from sending_msgs and add to sent_msgs
                self.sending_msgs.pop(msg_idx)
                msg_idx -= 1
                sent_msgs.append(msg)
                msg.end_time = self.current_time

            elif isclose(self.current_time - msg.last_sr_update, sr_update_period):
                if msg.send_rate < msg.dsg_send_rate:
                    msg.send_rate += msg.lgr * (self.current_time - msg.last_sr_update)
                if msg.send_rate > msg.dsg_send_rate:
                    # msg.send_rate -= msg.lgr * (self.current_time - msg.last_sr_update)
                    # Drop to dsr instantly
                    msg.send_rate = msg.dsg_send_rate
                    
                msg.last_sr_update = self.current_time

            msg_idx += 1
        
        timing.end()

        timing.start('process_sent')
        # Process sent msgs
        for msg in sent_msgs:

            # remove from active, update send rates
            self.sending[msg.from_id].remove(msg)
            self.receiving[msg.to_id].remove(msg)

            self.total_msgs -= 1
            
            self._update_dsg_send_rates()

        timing.end()

        timing.start('move_future')
        # Move future msgs in
        msg_idx = 0
        while msg_idx < len(self.future_msgs):
            msg = self.future_msgs[msg_idx]

            if msg.in_time == self.current_time:
                self.future_msgs.pop(msg_idx)
                msg_idx -= 1

                # Start time is always in time, because msgs start sending immediately
                msg.start_time = msg.in_time

                self.sending[msg.from_id].append(msg)
                self.receiving[msg.to_id].append(msg)
                self.sending_msgs.append(msg)
                self.total_msgs += 1

                self._update_dsg_send_rates()
                msg.send_rate = min(starting_sr, msg.dsg_send_rate)

            msg_idx += 1

        timing.end()

        # DSGSRs are accurate, calculate effective bandwidth
        if eff_start is not None:
            if self.current_time >= eff_start and self.current_time <= eff_end:

                # Get how much bw each node is using, in each direction
                for node_id in self.nodes:

                    # inbound
                    tsr = 0
                    for msg in self.sending_msgs:
                        if msg.to_id == node_id:
                            tsr += msg.dsg_send_rate

                    if len(self.eff_in[node_id]) == 0 or self.eff_in[node_id][-1][1] != tsr:
                        self.eff_in[node_id].append((self.current_time, tsr))

                    # outbound
                    tsr = 0
                    for msg in self.sending_msgs:
                        if msg.from_id == node_id:
                            tsr += msg.dsg_send_rate

                    if len(self.eff_out[node_id]) == 0 or self.eff_out[node_id][-1][1] != tsr:
                        self.eff_out[node_id].append((self.current_time, tsr))
                    



        # Return sent msgs
        return sent_msgs


if __name__ == '__main__':
    node_bws = ({
        0: 10,
        1: 10,
        2: 10,
        3: 10
    },
    {
        0: 10,
        1: 10,
        2: 10,
        3: 10
    })

    ne = NetworkEmulatorLite(node_bws, True)

    ne.send_msg(0, 1, 100, 0)
    ne.send_msg(0, 1, 100, 0)
    ne.send_msg(2, 0, 100, 0)
    # 2.5, 2.5, 5

    ne.send_msg(2, 0, 100, 10)
    # 2.5, 2.5, 2.5, 2.5

    ne.send_msg(1, 0, 100, 15)
    # 2.5, 2.5, 1.66, 1.66, 1.66

    ne.send_msg(1, 3, 100, 20)
    # 2.5, 2.5, 1.66, 1.66, 1.66, 3.33

    ne.send_msg(3, 1, 100, 25)

    msg = Message(0, 0, 20, 0)
    msg.lgr = 10
    msg.send_rate = 10

    print(ne._get_completion_time(msg))
    print(ne._predict_amt_sent(msg, 0.5))

    sent_msgs = []
    while len(sent_msgs) == 0:
        sent_msgs = ne.move()
        for msg in sent_msgs:
            print('from {0} to {1}, start {2}, end {3}'.format(msg.from_id, msg.to_id, msg.start_time, msg.end_time))
        sent_msgs = []