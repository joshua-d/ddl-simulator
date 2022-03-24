import random
import math
from enum import Enum


class MessageType(Enum):
    PARAMS = 0
    GRADS = 1


PARAMS_SIZE = 10
GRADS_SIZE = 20

class Message:
    def __init__(self, size, start_time, end_time, buffer, channel_id, msg_type):
        self.size = size
        self.start_time = start_time
        self.end_time = end_time
        self.buffer = buffer
        self.channel_id = channel_id
        self.type = msg_type
        
        self.needs_extending = False


class Channel:
    def __init__(self, id):
        self.id = id
        self.msgs = []
        self.time = 0
        self.next_msg_idx = 0

        # Type and size of next msg to be added
        self.prospective_msg_type = MessageType.PARAMS
        self.prospective_msg_size = PARAMS_SIZE

    def inc_pros_msg_type(self):
        if self.prospective_msg_type == MessageType.PARAMS:
            self.prospective_msg_type = MessageType.GRADS
            self.prospective_msg_size = GRADS_SIZE
        else:
            self.prospective_msg_type = MessageType.PARAMS
            self.prospective_msg_size = PARAMS_SIZE


class Bandwidth:

    def __init__(self, cluster, bandwidth):

        self.cluster = cluster
        self.bandwidth = bandwidth # B/s

        self.msg_ids = []
        self.msg_start_times = {}
        self.msg_end_times = {}

        # Channels by channel id
        self.channels = {}
        self._init_channels()

        # Constant fields

        self.min_msgs_per_channel = 2
        self.num_msgs_to_gen = 2

        self.min_work_delay = 0.010
        self.max_work_delay = 0.020

        self._init_msgs()


    def _init_channels(self):
        for worker in self.cluster.workers:
            for ps_id in self.cluster.parameter_servers:
                channel_id = 'w%sp%s' % (worker.id, ps_id)
                self.channels[channel_id] = Channel(channel_id)


    def _init_msgs(self):

        # Init each channel with first params msg
        for channel_id in self.channels:
            self._add_msg(channel_id, PARAMS_SIZE, MessageType.PARAMS)
            self.channels[channel_id].inc_pros_msg_type()

        # Add msgs until full

        least_msgs = 1

        while least_msgs < self.min_msgs_per_channel + self.num_msgs_to_gen:
            # Find channel with next earliest start time
            earliest_start_time = math.inf
            earliest_channel_id = None

            for channel_id in self.channels:
                channel = self.channels[channel_id]
                latest_msg = channel.msgs[-1]
                start_time = latest_msg.end_time + latest_msg.buffer
                if start_time < earliest_start_time:
                    earliest_start_time = start_time
                    earliest_channel_id = channel_id

            # Add msg to channel
            channel = self.channels[earliest_channel_id]
            msg_type = channel.prospective_msg_type
            msg_size = channel.prospective_msg_size
            channel.inc_pros_msg_type()

            self._add_msg(earliest_channel_id, msg_size, msg_type)

            # Check least msgs
            least_msgs = math.inf
            for channel_id in self.channels:
                if len(self.channels[channel_id].msgs) < least_msgs:
                    least_msgs = len(self.channels[channel_id].msgs)



    # Assumes all msgs are currently correct, time is between msg.start_time and msg.end_time
    def _get_amt_sent_at_time(self, msg, time):

        # Get change points
        change_points = []
        num_sending_msgs = 1

        for comp_channel_id in self.channels:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time > msg.start_time and comp_msg.start_time < time:
                    change_points.append((comp_msg.start_time, 's'))
                elif comp_msg.start_time <= msg.start_time and comp_msg.end_time > msg.start_time:
                    num_sending_msgs += 1
                    if comp_msg.end_time < time:
                        change_points.append((comp_msg.end_time, 'e'))

        change_points.append((time, 'f'))

        # Get amount sent
        last_change_time = msg.start_time
        amount_sent = 0
        current_bw = self.bandwidth / num_sending_msgs

        while len(change_points) > 0:
            current_bw = self.bandwidth / num_sending_msgs
            current_send_time = change_points[0][0] - last_change_time

            amount_sent += current_bw * current_send_time

            next_change_point = change_points.pop(0)
            if next_change_point[1] == 's':
                num_sending_msgs += 1
            else:
                num_sending_msgs -= 1

            last_change_time = next_change_point[0]


        return amount_sent


    # Assumes that this msg's new end time will not extend past any other msg's prospective end time
    # Called on overlapping msgs in order from least end time to greatest end time
    # Doesn't consider end points of msgs that still need extending (needs_extending) -
    #   if a msg needs_extending, it's prospective end point is further than this one's, so its end point need not be
    #   considered as a change point
    def _extend_msg_end_time(self, msg, new_msg_start_time):

        # TODO could probably benefit from some sort of start time mapping here !!!!
        # TODO could combine these 2 sections into one with logic similar to get_amt

        # Count initial current sending msgs at new_msg_start_time
        num_sending_msgs = 2 # 1 for this msg, 1 for new msg

        for comp_channel_id in self.channels:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time <= new_msg_start_time and comp_msg.end_time > new_msg_start_time:
                    num_sending_msgs += 1


        # Get change points
        change_points = []

        for comp_channel_id in self.channels:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time > new_msg_start_time:
                    change_points.append((comp_msg.start_time, 's'))

                if comp_msg.end_time > new_msg_start_time and not comp_msg.needs_extending:
                    change_points.append((comp_msg.end_time, 'e'))


        # extend
        last_change_time = new_msg_start_time
        amount_sent = self._get_amt_sent_at_time(msg, new_msg_start_time)
        current_bw = self.bandwidth / num_sending_msgs

        while len(change_points) > 0:
            current_bw = self.bandwidth / num_sending_msgs
            current_send_time = change_points[0][0] - last_change_time

            amount_able_to_send = current_bw * current_send_time

            if amount_sent + amount_able_to_send > msg.size:
                # reached an end point before next change point
                break

            else:
                amount_sent += amount_able_to_send

                next_change_point = change_points.pop(0)
                if next_change_point[1] == 's':
                    num_sending_msgs += 1
                else:
                    num_sending_msgs -= 1

                last_change_time = next_change_point[0]

        # no change points left, still some to send
        current_bw = self.bandwidth / num_sending_msgs
        amount_left = msg.size - amount_sent
        time_left = amount_left / current_bw
        new_end_time = last_change_time + time_left

        msg.end_time = new_end_time
        msg.needs_extending = False


    # Assumes that all other messages are correct based on this msg having end time = inf
    # TODO this is basically the same as extend...
    def _set_msg_end_time(self, msg):

        # Count initial current sending msgs at start time
        num_sending_msgs = 1 # 1 for this msg

        for comp_channel_id in self.channels:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time <= msg.start_time and comp_msg.end_time > msg.start_time:
                    num_sending_msgs += 1


        # Get change points
        change_points = []

        for comp_channel_id in self.channels:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time > msg.start_time:
                    change_points.append((comp_msg.start_time, 's'))

                if comp_msg.end_time > msg.start_time:
                    change_points.append((comp_msg.end_time, 'e'))


        # calc end time
        last_change_time = msg.start_time
        amount_sent = 0
        current_bw = self.bandwidth / num_sending_msgs

        while len(change_points) > 0:
            current_bw = self.bandwidth / num_sending_msgs
            current_send_time = change_points[0][0] - last_change_time

            amount_able_to_send = current_bw * current_send_time

            if amount_sent + amount_able_to_send > msg.size:
                # reached an end point before next change point
                break

            else:
                amount_sent += amount_able_to_send

                next_change_point = change_points.pop(0)
                if next_change_point[1] == 's':
                    num_sending_msgs += 1
                else:
                    num_sending_msgs -= 1

                last_change_time = next_change_point[0]

        # no change points left, still some to send
        current_bw = self.bandwidth / num_sending_msgs
        amount_left = msg.size - amount_sent
        time_left = amount_left / current_bw
        new_end_time = last_change_time + time_left
        msg.end_time = new_end_time


    # Assumes all messages already in self.msgs already have valid start and end time
    # Assumes this message will have the next earliest start time of all channels - must add msgs in this order!
    # Buffer is the delay between the end time of this msg and the start time of the next one
    def _add_msg(self, channel_id, size, msg_type):
        channel = self.channels[channel_id]
        channel_msgs = channel.msgs

        # Set added msg's start time
        if len(channel_msgs) > 0:
            start_time = channel_msgs[-1].end_time + channel_msgs[-1].buffer
        else:
            start_time = 0


        # Set added msg's buffer
        if msg_type == MessageType.PARAMS:
            buffer = random.uniform(self.min_work_delay, self.max_work_delay)
        else:
            buffer = 0 # TODO consider adding work delay for PS to do grads

        added_msg = Message(size, start_time, math.inf, buffer, channel_id, msg_type)


        # Get overlapping messages
        overlapping_msgs = []

        for comp_channel_id in self.channels:
            if comp_channel_id == channel_id:
                continue

            for comp_msg in self.channels[comp_channel_id].msgs:
                if comp_msg.start_time <= added_msg.start_time and comp_msg.end_time > added_msg.start_time:  # comp msg is sending when this one starts
                    overlapping_msgs.append(comp_msg)
                    comp_msg.needs_extending = True


        # Correct end times of overlapping messages
        overlapping_msgs.sort(key=lambda e: e.end_time)

        for overlapping_msg in overlapping_msgs:
            self._extend_msg_end_time(overlapping_msg, start_time)


        # Set added msg's real end time
        self._set_msg_end_time(added_msg)

        # add added_msg to channel
        self.channels[channel_id].msgs.append(added_msg)

        # Correct end times of msgs that end after real end time
        # TODO I think I can just use _set_msg_end_time as is?
        for overlapping_msg in overlapping_msgs:
            if overlapping_msg.end_time > added_msg.end_time:
                self._set_msg_end_time(overlapping_msg)





    def _get_channel_id(self, wk_id, ps_id):
        return 'w%sp%s' % (wk_id, ps_id)


    # Removes msgs before next_msg_idx of each channel and resets next_msg_idx
    def remove_used_msgs(self):
        for channel_id in self.channels:
            channel = self.channels[channel_id]
            channel.msgs = channel.msgs[channel.next_msg_idx:]
            channel.next_msg_idx = 0


    # Adds more messages up to the min channel having min_msgs_per_channel + num_msgs_to_gen msgs
    # Calls remove_used_msgs when done to clean out old msgs
    def _prepare_msgs(self):
        # Add msgs until full

        least_msgs = 0

        while least_msgs < self.min_msgs_per_channel + self.num_msgs_to_gen:
            # Find channel with next earliest start time
            earliest_start_time = math.inf
            earliest_channel_id = None

            for channel_id in self.channels:
                channel = self.channels[channel_id]
                latest_msg = channel.msgs[-1]
                start_time = latest_msg.end_time + latest_msg.buffer
                if start_time < earliest_start_time:
                    earliest_start_time = start_time
                    earliest_channel_id = channel_id

            # Add msg to channel
            channel = self.channels[earliest_channel_id]
            msg_type = channel.prospective_msg_type
            msg_size = channel.prospective_msg_size
            channel.inc_pros_msg_type()

            self._add_msg(earliest_channel_id, msg_size, msg_type)

            # Check least msgs
            least_msgs = math.inf
            for channel_id in self.channels:
                if len(self.channels[channel_id].msgs) < least_msgs:
                    len(self.channels[channel_id].msgs)

        self.remove_used_msgs()


    # Called by thread when sending point is reached - updates state of the data structure
    def send_msg(self, wk_id, ps_id):
        channel_id = self._get_channel_id(wk_id, ps_id)
        channel = self.channels[channel_id]

        msg_end_time = channel.msgs[channel.next_msg_idx].end_time

        wait_time = msg_end_time - channel.time

        channel.time = msg_end_time
        channel.next_msg_idx += 1 # TODO maybe mark message as used

        # Check if there are enough msgs left
        if len(channel.msgs) - channel.next_msg_idx < self.min_msgs_per_channel:
            self._prepare_msgs()

        return wait_time





class TestWorker:
    def __init__(self, id):
        self.id = id

class TestCluster:
    def __init__(self, workers, ps_ids):
        self.workers = workers
        self.parameter_servers = ps_ids


wks = []
for i in range(4):
    wks.append(TestWorker(i))

ps_ids = []
for i in range(1):
    ps_ids.append(i)

cl = TestCluster(wks, ps_ids)

bw = Bandwidth(cl, 10)





for channel_id in bw.channels:
    print('Channel %s' % channel_id)
    for msg in bw.channels[channel_id].msgs:
        print('start: %f, end: %f' % (msg.start_time, msg.end_time))
    print()
