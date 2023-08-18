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

        # used in NEL.move
        self.prospective_amt_sent = 0

        self.time_to_reach_dsr = 0
        self.amt_sent_at_dsr_reach = 0 # only valid if dsr has not been reached


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
                        if msg.send_rate > msg.dsg_send_rate:
                            msg.send_rate = msg.dsg_send_rate # SR drops to DSR instantly
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
                        if msg.send_rate > msg.dsg_send_rate:
                            msg.send_rate = msg.dsg_send_rate # SR drops to DSR instantly
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

        # Called whenever DSRs are updated
        self._compute_sr_info()


    def _compute_sr_info(self):
        for msg in self.sending_msgs:
            msg.time_to_reach_dsr = (msg.dsg_send_rate - msg.send_rate) / msg.lgr

            t = msg.time_to_reach_dsr
            msg.amt_sent_at_dsr_reach = msg.amt_sent + msg.send_rate * t + msg.lgr * t * t


    def send_msg(self, from_id, to_id, msg_size, in_time):
        msg = Message(from_id, to_id, msg_size, in_time)

        self.future_msgs.append(msg)


    # Time is absolute, a future 'current_time' value
    def _predict_amt_sent(self, msg, time):
        t = time - self.current_time

        # Requested time is during rampup
        if t < msg.time_to_reach_dsr:
            return msg.amt_sent + msg.send_rate * t + msg.lgr * t * t
        
        # Msg has already reached dsr
        elif msg.time_to_reach_dsr == 0:
            return msg.amt_sent + msg.send_rate * t
        
        # Requested time is after rampup
        else:
            return msg.amt_sent_at_dsr_reach + msg.dsg_send_rate * (t - msg.time_to_reach_dsr)
    

    # Returns an absolute time
    def _get_completion_time(self, msg):

        # Msg has already reached dsr
        if msg.time_to_reach_dsr == 0:
            return (msg.size - msg.amt_sent) / msg.send_rate + self.current_time
        
        # Msg will complete during rampup
        if msg.amt_sent_at_dsr_reach > msg.size:
            a = msg.lgr
            b = msg.send_rate
            c = -(msg.size - msg.amt_sent)

            discriminant = b**2 - 4*a*c
            x1 = (-b + sqrt(discriminant)) / (2*a)
            x2 = (-b - sqrt(discriminant)) / (2*a)
            return max(x1, x2) + self.current_time
        
        # Msg will complete after rampup
        else:
            return msg.time_to_reach_dsr + (msg.size - msg.amt_sent_at_dsr_reach) / msg.dsg_send_rate + self.current_time


    # TODO with this new impl, 1 or 0 msgs send per move - may or may not be more efficient than using isclose
    # same with bringing future msgs in
    # TODO make sure using inf does not take a lot of time
    # TODO: eff_start cannot be 0
    def move(self, eff_start=None, eff_end=None):

        # Find next earliest in time
        earliest_in_time = inf
        next_in_msg = None

        for msg in self.future_msgs:
            if msg.in_time < earliest_in_time:
                earliest_in_time = msg.in_time
                next_in_msg = msg

        # Get amt sent at next earliest in time for each msg, and check for completion
        earliest_completion_time = inf
        next_completed_msg = None

        for msg in self.sending_msgs:
            msg.prospective_amt_sent = self._predict_amt_sent(msg, earliest_in_time)

            if msg.prospective_amt_sent > msg.size:
                comp_time = self._get_completion_time(msg)
                if comp_time < earliest_completion_time:
                    earliest_completion_time = comp_time
                    next_completed_msg = msg


        # Write prospective amt sents, update SRs, and move to in time
        if next_completed_msg is None:

            for msg in self.sending_msgs:
                msg.amt_sent = msg.prospective_amt_sent
                msg.send_rate = max(msg.send_rate + msg.lgr * (earliest_in_time - self.current_time), msg.dsg_send_rate)

            self.current_time = earliest_in_time

            # move msg in
            self.future_msgs.remove(next_in_msg)

            next_in_msg.start_time = next_in_msg.in_time    # Start time is always in time, because msgs start sending immediately

            self.sending[next_in_msg.from_id].append(next_in_msg)
            self.receiving[next_in_msg.to_id].append(next_in_msg)
            self.sending_msgs.append(next_in_msg)
            self.total_msgs += 1

            next_in_msg.send_rate = min(starting_sr, next_in_msg.dsg_send_rate)
            self._update_dsg_send_rates()
            

        # Update amt sents and SRs and move to completion time
        else:
            msg_idx = 0
            while msg_idx < len(self.sending_msgs):
                msg = self.sending_msgs[msg_idx]

                if msg == next_completed_msg:
                    # message has sent, remove from sending_msgs and add to sent_msgs
                    self.sending_msgs.pop(msg_idx)
                    msg_idx -= 1
                    msg.end_time = earliest_completion_time

                else:
                    msg.amt_sent = self._predict_amt_sent(msg, earliest_completion_time)
                    msg.send_rate = max(msg.send_rate + msg.lgr * (earliest_in_time - self.current_time), msg.dsg_send_rate)

                msg_idx += 1

            # Process sent
            # remove from active, update send rates
            self.sending[next_completed_msg.from_id].remove(next_completed_msg)
            self.receiving[next_completed_msg.to_id].remove(next_completed_msg)

            self.total_msgs -= 1
            
            self._update_dsg_send_rates()

            # Advance current time
            self.current_time = earliest_completion_time


        # Return sent msgs
        return [next_completed_msg]


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