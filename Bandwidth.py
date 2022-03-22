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
        
        self.needs_extending = False


class Channel:
    def __init__(self, id):
        self.id = id
        self.msgs = []
        self.time = 0
        self.next_msg_idx = 0


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


    def _init_channels(self):
        for worker in self.cluster.workers:
            for ps_id in self.cluster.parameter_servers:
                channel_id = 'w%sp%s' % (worker.id, ps_id)
                self.channels[channel_id] = Channel(channel_id)


    # Assumes all msgs are currently correct, time is between msg.start_time and msg.end_time
    def get_amt_sent_at_time(self, msg, time):

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
    def extend_msg_end_time(self, msg, new_msg_start_time):

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
        amount_sent = self.get_amt_sent_at_time(msg, new_msg_start_time)
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
    def set_msg_end_time(self, msg):

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
    def add_msg(self, channel_id, size):
        channel_msgs = self.channels[channel_id].msgs

        if len(channel_msgs) > 0:
            start_time = channel_msgs[-1].end_time  # TODO consider adding buffer here
            start_time += 1
        else:
            # start_time = random.uniform(0, 0.100)  # TODO edit rand start time
            start_time = 0

        added_msg = Message(size, start_time, math.inf, channel_id)


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
            self.extend_msg_end_time(overlapping_msg, start_time)


        # Set added msg's real end time
        self.set_msg_end_time(added_msg)

        # add added_msg to channel
        self.channels[channel_id].msgs.append(added_msg)

        # Correct end times of msgs that end after real end time
        # TODO I think I can just use set_msg_end_time as is?
        for overlapping_msg in overlapping_msgs:
            if overlapping_msg.end_time > added_msg.end_time:
                self.set_msg_end_time(overlapping_msg)


    def prepare_next_msgs(self, num_msgs):
        pass


    def get_channel_id(self, wk_id, ps_id):
        return 'w%sp%s' % (wk_id, ps_id)


    def send_msg(self, wk_id, ps_id):
        channel_id = self.get_channel_id(wk_id, ps_id)
        channel = self.channels[channel_id]

        msg_end_time = channel.msgs[channel.next_msg_idx].end_time

        wait_time = msg_end_time - channel.time

        channel.time = msg_end_time
        channel.next_msg_idx += 1 # TODO maybe mark message as used

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

bw.add_msg('w0p0', 5)
bw.add_msg('w1p0', 5)
bw.add_msg('w2p0', 5)
bw.add_msg('w0p0', 10)



for channel_id in bw.channels:
    print('Channel %s' % channel_id)
    for msg in bw.channels[channel_id].msgs:
        print('start: %f, end: %f' % (msg.start_time, msg.end_time))
    print()
