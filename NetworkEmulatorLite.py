from math import inf


class Message:
    def __init__(self, from_id, to_id, size, in_time, last_checked, send_rate, msg_id):
        self.from_id = from_id
        self.to_id = to_id
        self.size = size

        self.amt_sent = 0
        self.in_time = in_time
        self.last_checked = last_checked

        self.send_rate = send_rate

        self.id = msg_id


class NetworkEmulatorLite:

    def __init__(self, node_bws):

        self.inbound_max, self.outbound_max = node_bws

        self.queued_msgs = {}
        self.active_msgs = {}
        self.receiving = {} # true if all msgs in active are incoming or active msgs is empty

        for node_id in self.inbound_max.keys():
            self.queued_msgs[node_id] = []
            self.active_msgs[node_id] = []
            self.receiving[node_id] = True


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
            if self.receiving[node_id]:
                for msg in self.active_msgs[node_id]:
                    incoming_offering[msg] = self.inbound_max[node_id] / len(self.active_msgs[node_id])
            else:
                for msg in self.active_msgs[node_id]:
                    outgoing_offering[msg] = self.outbound_max[node_id] / len(self.active_msgs[node_id])

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
                for msg in self.active_msgs[least_offering_node_id]:
                    if msg not in final_msgs:

                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.active_msgs[msg.from_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (outgoing_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                outgoing_offering[aux_msg] += distribute_amt
                    
            else:
                for msg in self.active_msgs[least_offering_node_id]:
                    if msg not in final_msgs:

                        msg.send_rate = least_offering
                        final_msgs.append(msg)
                        
                        distribute_msgs = []
                        for aux_msg in self.active_msgs[msg.to_id]:
                            if aux_msg not in final_msgs:
                                distribute_msgs.append(aux_msg)

                        if len(distribute_msgs) != 0:
                            distribute_amt = (incoming_offering[msg] - least_offering) / len(distribute_msgs)

                            for aux_msg in distribute_msgs:
                                incoming_offering[aux_msg] += distribute_amt


    def send_msg(self, from_id, to_id, msg_size, in_time, msg_id):
        msg = Message(from_id, to_id, msg_size, in_time, in_time, 0, msg_id)

        self.future_msgs.append(msg)


        


    def _ready_to_receive(self, from_id, to_id):

        if not self.receiving[to_id] or len(self.queued_msgs[to_id]) == 0: # TODO second case should never happen
            return False

        queued = self.queued_msgs[to_id]

        queue_idx = 0
        while queue_idx < len(queued) and queued[queue_idx].to_id == to_id:
            queued_msg = queued[queue_idx]
            if queued_msg.from_id == from_id:
                return True
            queue_idx += 1

        return False


    def _ready_to_send(self, from_id, to_id):

        if self.receiving and len(self.active_msgs[from_id]) != 0 or len(self.queued_msgs[from_id]) == 0: # TODO second case should never happen
            return False

        queued = self.queued_msgs[from_id]

        queue_idx = 0
        while queue_idx < len(queued) and queued[queue_idx].from_id == from_id:
            queued_msg = queued[queue_idx]
            if queued_msg.to_id == to_id:
                return True
            queue_idx += 1

        return False


    def _check_queue(self, node_id, check_time):
        queued = self.queued_msgs[node_id]
        active = self.active_msgs[node_id]
        receiving = self.receiving[node_id]

        if len(queued) == 0:
            return

        if queued[0].from_id == node_id:
            # next msgs are outgoing

            # check that node is ready to send
            if receiving and len(active) != 0:
                return

            # while next msg in queue is outgoing
            queue_idx = 0
            while queue_idx < len(queued) and queued[queue_idx].from_id == node_id:

                # prepare utils
                queued_msg = queued[queue_idx]
                to_node_queued = self.queued_msgs[queued_msg.to_id]
                to_node_active = self.active_msgs[queued_msg.to_id]

                if self._ready_to_receive(node_id, queued_msg.to_id):
                    # incoming node is ready to receive

                    # remove msg from queue, move into active
                    to_node_queued.remove(queued_msg)
                    to_node_active.append(queued_msg)
                    queued.remove(queued_msg)
                    active.append(queued_msg)

                    # update from-node's mode
                    self.receiving[node_id] = False

                    # set entry time and move into sending_msgs
                    queued_msg.last_checked = check_time
                    self.sending_msgs.append(queued_msg)
                    
                    self.total_msgs += 1

                else:
                    # incoming node is not ready to receive
                    queue_idx += 1

        else:
            # next msgs are incoming

            # check that node is ready to receive
            if not receiving:
                return

            # while next msg in queue is incoming
            queue_idx = 0
            while queue_idx < len(queued) and queued[queue_idx].to_id == node_id:

                # prepare utils
                queued_msg = queued[queue_idx]
                from_node_queued = self.queued_msgs[queued_msg.from_id]
                from_node_active = self.active_msgs[queued_msg.from_id]

                if self._ready_to_send(queued_msg.from_id, node_id):
                    # outgoing node is ready to send

                    # remove msg from queue, move into active
                    from_node_queued.remove(queued_msg)
                    from_node_active.append(queued_msg)
                    queued.remove(queued_msg)
                    active.append(queued_msg)

                    # update modes
                    self.receiving[node_id] = True
                    self.receiving[queued_msg.from_id] = False

                    # set entry time and move into sending_msgs
                    queued_msg.last_checked = check_time
                    self.sending_msgs.append(queued_msg)
                    
                    self.total_msgs += 1

                else:
                    # outgoing node is not ready to send
                    queue_idx += 1


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

            if msg.amt_sent >= msg.size:
                # message has sent, remove from sending_msgs and add to sent_msgs
                self.sending_msgs.pop(msg_idx)
                msg_idx -= 1
                sent_msgs.append(msg)

            msg_idx += 1

        for msg in sent_msgs:

            # remove from active, pull next from queues, update send rates
            from_active = self.active_msgs[msg.from_id]
            to_active = self.active_msgs[msg.to_id]

            from_active.remove(msg)
            to_active.remove(msg)

            self.total_msgs -= 1

            if len(from_active) == 0:
                self.receiving[msg.from_id] = True
                self._check_queue(msg.from_id, self.current_time)

            if len(to_active) == 0:
                self._check_queue(msg.to_id, self.current_time)

            final_msg_send_rate = msg.send_rate

            self._update_send_rates()

        # Move future msgs in
        msg_idx = 0
        while msg_idx < len(self.future_msgs):
            msg = self.future_msgs[msg_idx]

            if msg.in_time == self.current_time:
                self.future_msgs.pop(msg_idx)
                msg_idx -= 1

                self.queued_msgs[msg.from_id].append(msg)
                self.queued_msgs[msg.to_id].append(msg)

                self._check_queue(msg.from_id, self.current_time)
                self._update_send_rates()

            msg_idx += 1

        # Return sent msgs
        return sent_msgs
