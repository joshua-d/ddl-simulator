import random
import math


wk_msg_size = 20
ps_msg_size = 10


class Message:
    def __init__(self, size, start_time, end_time, channel_id):
        self.size = size
        self.start_time = start_time
        self.end_time = end_time
        self.channel_id = channel_id


class Bandwidth:

    def __init__(self, cluster, bandwidth):

        self.cluster = cluster
        self.bandwidth = bandwidth # B/s

        self.msg_ids = []
        self.msg_start_times = {}
        self.msg_end_times = {}


        # List of Messages by channel ID
        self.msgs = {}

        self._init_msgs()

    def send_gradients(self, wk_id, ps_id, bytes, callback):
        pass

    def send_params(self, ps_id, wk_id, bytes, callback):
        pass


    def _init_msgs(self):
        pass



    # Assumes all msgs are currently correct, time is between msg.start_time and msg.end_time
    def get_amt_sent_at_time(self, msg, time):

        # Get overlapping msgs with end times after time
        overlapping_msgs = []

        for comp_channel_id in self.msgs:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.msgs[comp_channel_id]:
                if comp_msg.end_time > time and comp_msg.start_time < msg.end_time:
                    overlapping_msgs.append(comp_msg)

        # Get points where bw changes and bandwidth at time
        change_points = []
        num_sending_msgs = 0

        for comp_msg in overlapping_msgs:
            if comp_msg.start_time > time:
                change_points.append((comp_msg.start_time, 's'))
            else:
                num_sending_msgs += 1

            if comp_msg.end_time < msg.end_time:
                change_points.append((comp_msg.end_time, 'e'))

        change_points.append((msg.end_time, 'f'))

        # Get amount sent
        amount_sent = 0
        while len(change_points) > 0:
            current_bw = self.bandwidth / num_sending_msgs
            current_send_time = change_points[0][0] - time

            amount_sent += current_bw * current_send_time

            if change_points.pop(0)[1] == 's':
                num_sending_msgs += 1
            else:
                num_sending_msgs -= 1


        return msg.size - amount_sent


        
        


    # Assumes all messages already in self.msgs already have valid start and end time
    def add_msg(self, channel_id, size):
        channel_msgs = self.msgs[channel_id]

        if len(channel_msgs) > 0:
            start_time = channel_msgs[-1].end_time  # TODO consider adding buffer here
        else:
            start_time = random.uniform(0, 0.100)  # TODO edit rand start time

        added_msg = Message(size, start_time, math.inf)
        # channel_msgs.append(added_msg)


        # Get overlapping messages
        overlapping_msgs = []

        for comp_channel_id in self.msgs:
            if comp_channel_id == channel_id:
                continue

            for comp_msg in self.msgs[comp_channel_id]:
                if comp_msg.end_time > added_msg.start_time:  # comp msg ends before this one starts
                    overlapping_msgs.append(comp_msg)


        # Correct end times of overlapping messages
        for overlapping_msg in overlapping_msgs:
            self.set_msg_end_time(overlapping_msg)


        
