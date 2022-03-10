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
        # for worker in self.cluster.workers:
        #     for ps_id in self.cluster.parameter_servers:
        #         self.msgs['w%sp%s' % (worker.id, ps_id)] = []



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
        num_sending_msgs = 1

        for comp_msg in overlapping_msgs:
            if comp_msg.start_time > time:
                change_points.append((comp_msg.start_time, 's'))
            else:
                num_sending_msgs += 1

            if comp_msg.end_time < msg.end_time:
                change_points.append((comp_msg.end_time, 'e'))

        change_points.append((msg.end_time, 'f'))

        # Get amount sent
        last_change_time = time # time at beginning of change point
        amount_sent = 0

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



        return msg.size - amount_sent


    # Assumes that this msg's new end time will not extend past any other msg's prospective end time
    # Called on overlapping msgs in order from least end time to greatest end time
    # Considers only starts, not ends
    # TODO don't forget to push msgs in this channel !!!!

    # Assumes that ends before this msg's end are final, but ignores ends after this end time
    def extend_msg_end_time(self, msg, new_msg_start_time):
        amount_sent = self.get_amt_sent_at_time(msg, new_msg_start_time)

        # TODO could probably benefit from some sort of start time mapping here !!!!

        # Count initial current sending msgs at new_msg_start_time
        num_sending_msgs = 2 # 1 for this msg, 1 for new msg

        for comp_channel_id in self.msgs:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.msgs[comp_channel_id]:
                if comp_msg.start_time <= new_msg_start_time and comp_msg.end_time > new_msg_start_time:
                    num_sending_msgs += 1


        # Get change points
        change_points = []

        for comp_channel_id in self.msgs:
            if comp_channel_id == msg.channel_id:
                continue

            for comp_msg in self.msgs[comp_channel_id]:
                if comp_msg.start_time > new_msg_start_time:
                    change_points.append((comp_msg.start_time, 's'))

                if comp_msg.end_time > new_msg_start_time and comp_msg.end_time < msg.end_time:
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
        amount_left = msg.size - amount_sent
        time_left = amount_left / current_bw
        new_end_time = last_change_time + time_left
        msg.end_time = new_end_time
        # TODO push back msgs in channel hereeeee !!!
        return
        



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
        overlapping_msgs.sort(key=lambda e: e.end_time)

        for overlapping_msg in overlapping_msgs:
            self.extend_msg_end_time(overlapping_msg, start_time)


        


bw = Bandwidth(None, 10)

bw.msgs[0] = [] # channel ids
bw.msgs[1] = []

m1 = Message(10, 0, 1, 0)
bw.msgs[0].append(m1)

# m2 = Message(10, 0, 1.5, 1)
# bw.msgs[1].append(m2)


bw.extend_msg_end_time(m1, 0.75)

print(m1.end_time)