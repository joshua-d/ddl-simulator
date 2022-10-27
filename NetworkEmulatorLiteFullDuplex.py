from math import inf, isclose


class Message:
    def __init__(self, from_id, to_id, size, in_time, last_checked, send_rate):
        self.from_id = from_id
        self.to_id = to_id
        self.size = size

        self.amt_sent = 0
        self.in_time = in_time
        self.last_checked = last_checked

        self.send_rate = send_rate

        self.start_time = 0
        self.end_time = 0


class NetworkEmulatorLiteFullDuplex:

    def __init__(self, node_bws):

        self.inbound_max, self.outbound_max = node_bws

        self.sending = {}
        self.receiving = {}

        for node_id in self.inbound_max.keys():
            self.sending[node_id] = []
            self.receiving[node_id] = []

        self.total_msgs = 0  # TODO could perhaps use len(self.sending_msgs)

        # Messages currently using bandwidth
        self.sending_msgs = []

        self.current_time = 0

        self.future_msgs = []


    def _update_send_rates(self):

        # Prepare
        incoming_offering = {}
        outgoing_offering = {}

        for node_id in self.inbound_max.keys():
            for msg in self.sending[node_id]:
                incoming_offering[msg] = self.inbound_max[node_id] / len(self.sending[node_id])

                # This part simulates half duplex
                if len(self.receiving[node_id]) != 0:
                    incoming_offering[msg] /= 2

            for msg in self.receiving[node_id]:
                outgoing_offering[msg] = self.outbound_max[node_id] / len(self.receiving[node_id])

                # This part simulates half duplex
                if len(self.sending[node_id]) != 0:
                    outgoing_offering[msg] /= 2

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

                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.receiving[msg.from_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (outgoing_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                outgoing_offering[aux_msg] += distribute_amt
                    
            else:
                for msg in self.sending[least_offering_node_id]:
                    if msg not in final_msgs:

                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.sending[msg.to_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (incoming_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                incoming_offering[aux_msg] += distribute_amt


    def send_msg(self, from_id, to_id, msg_size, in_time):
        msg = Message(from_id, to_id, msg_size, in_time, in_time, 0)

        self.future_msgs.append(msg)


    def move(self):

        # Find next earliest completion time
        earliest_completion_time = inf

        for msg in self.sending_msgs:
            msg_completion_time = self.current_time + (msg.size - msg.amt_sent) / msg.send_rate
            if msg_completion_time < earliest_completion_time:
                earliest_completion_time = msg_completion_time

        # Find next earliest in time
        earliest_in_time = inf

        for msg in self.future_msgs:
            if msg.in_time < earliest_in_time:
                earliest_in_time = msg.in_time

        # Move to next earliest completion time or in time
        self.current_time = min(earliest_completion_time, earliest_in_time)

        # Process sending msgs
        sent_msgs = []

        msg_idx = 0
        while msg_idx < len(self.sending_msgs):
            msg = self.sending_msgs[msg_idx]

            msg.amt_sent += (self.current_time - msg.last_checked) * msg.send_rate
            msg.last_checked = self.current_time

            if msg.amt_sent > msg.size or isclose(msg.amt_sent, msg.size):
                # message has sent, remove from sending_msgs and add to sent_msgs
                self.sending_msgs.pop(msg_idx)
                msg_idx -= 1
                sent_msgs.append(msg)
                msg.end_time = self.current_time

            msg_idx += 1

        # Process sent msgs
        for msg in sent_msgs:

            # remove from active, update send rates
            self.sending[msg.from_id].remove(msg)
            self.receiving[msg.to_id].remove(msg)

            self.total_msgs -= 1
            
            self._update_send_rates()

        # Move future msgs in
        msg_idx = 0
        while msg_idx < len(self.future_msgs):
            msg = self.future_msgs[msg_idx]

            if msg.in_time == self.current_time:
                self.future_msgs.pop(msg_idx)
                msg_idx -= 1

                # Start time is always in time, because with FD msgs start sending immediately
                msg.start_time = msg.in_time

                self.sending[msg.from_id].append(msg)
                self.receiving[msg.to_id].append(msg)
                self.sending_msgs.append(msg)
                self.total_msgs += 1

                self._update_send_rates()

            msg_idx += 1

        # Return sent msgs
        return sent_msgs
