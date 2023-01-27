import datetime, random
from NetworkEmulatorLite import NetworkEmulatorLite
import json


class Event:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


class WorkerStepEvent(Event):
    def __init__(self, start_time, end_time, worker_id):
        super().__init__(start_time, end_time)
        self.worker_id = worker_id


class SendParamsEvent(Event):
    def __init__(self, start_time, end_time, sender_id, receiver_id):
        super().__init__(start_time, end_time)
        self.sender_id = sender_id
        self.receiver_id = receiver_id


class ReceiveParamsEvent(Event):
    def __init__(self, start_time, end_time, sender_id, receiver_id):
        super().__init__(start_time, end_time)
        self.sender_id = sender_id
        self.receiver_id = receiver_id

        
class PSAggrEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSApplyEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSParentApplyEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id



class Worker:
    def __init__(self, id, step_time, st_variation):
        self.id = id
        self.parent = None

        self.step_time = step_time
        self.st_variation = st_variation


class ParameterServer:
    def __init__(self, id, sync_style, aggr_time, apply_time):
        self.id = id
        self.parent = None
        self.children = []

        self.sync_style = sync_style
        self.aggr_time = aggr_time
        self.apply_time = apply_time

        self.n_param_sets_received = 0

        self.waiting_for_parent = False
        self.child_msg_queue = []
        self.waiting_child_id = None

        self.next_available_work_time = 0


class NetworkSequenceGenerator:

    def __init__(self, node_descs, msg_size, half_duplex):

        # In bits
        self.msg_size = msg_size

        self.nodes = []
        self.workers = []
        self.parameter_servers = []

        self.events = []

        inbound_max = {}
        outbound_max = {}

        # Build node objs from config
        for node_desc in node_descs:
            if node_desc['node_type'] == 'ps':
                ps = ParameterServer(
                        node_desc['id'],
                        node_desc['sync_style'],
                        node_desc['aggr_time'],
                        node_desc['apply_time']
                    )
                self.nodes.append(ps)
                self.parameter_servers.append(ps)
                inbound_max[ps.id] = node_desc['inbound_bw'] * 1000000
                outbound_max[ps.id] = node_desc['outbound_bw'] * 1000000

            elif node_desc['node_type'] == 'worker':
                w = Worker(
                        node_desc['id'],
                        node_desc['step_time'],
                        node_desc['st_variation']
                    )
                self.nodes.append(w)
                self.workers.append(w)
                inbound_max[w.id] = node_desc['inbound_bw'] * 1000000
                outbound_max[w.id] = node_desc['outbound_bw'] * 1000000

        # Nodes in config must be in order of ID
        # Build parents and children lists
        for node_desc in node_descs:
            node = self.nodes[node_desc['id']]
            if node_desc['parent'] is not None:
                node.parent = self.nodes[node_desc['parent']]
                node.parent.children.append(node)

        # Build NE
        self.ne = NetworkEmulatorLite((inbound_max, outbound_max), half_duplex)

        # Set up starting events
        for worker in self.workers:
            step_time = worker.step_time - worker.st_variation + random.uniform(0, worker.st_variation*2)
            self.events.append(WorkerStepEvent(0, step_time, worker.id))
            self.events.append(SendParamsEvent(step_time, step_time, worker.id, worker.parent.id))
            self.ne.send_msg(worker.id, worker.parent.id, self.msg_size, step_time)

        

    def _process_msg(self, msg):

        if type(self.nodes[msg.to_id]) == ParameterServer:

            ps = self.nodes[msg.to_id]

            if ps.waiting_for_parent:
                if msg.from_id == ps.parent.id:
                    # Handle update from parent

                    self.events.append(ReceiveParamsEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id))

                    # We're adding a zero time apply to make event sequence more useful, see below in async relay too TODO
                    self.events.append(PSParentApplyEvent(msg.end_time, msg.end_time, ps.id))

                    # Immediately send to child(ren)
                    if ps.sync_style == 'async':
                        self.events.append(SendParamsEvent(self.ne.current_time, self.ne.current_time, ps.id, ps.waiting_child_id))
                        self.ne.send_msg(ps.id, ps.waiting_child_id, self.msg_size, self.ne.current_time)
                    elif ps.sync_style == 'sync':
                        for child in ps.children:
                            self.events.append(SendParamsEvent(self.ne.current_time, self.ne.current_time, ps.id, child.id))
                            self.ne.send_msg(ps.id, child.id, self.msg_size, self.ne.current_time)

                    ps.waiting_for_parent = False

                    # Process msgs from children
                    if ps.sync_style == 'sync':
                        for child_msg in ps.child_msg_queue:
                            self._process_msg(child_msg)

                    elif ps.sync_style == 'async' and len(ps.child_msg_queue) != 0:
                        self._process_msg(ps.child_msg_queue.pop(0))
                    
                else:
                    # Got update from child while waiting for parent
                    ps.child_msg_queue.append(msg)

            else:
                # Not waiting for parent, update is from child

                if ps.sync_style == 'async':
                    # Process params immediately

                    # Add receive event
                    self.events.append(ReceiveParamsEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id))
                    
                    # If this is a mid level ps, send up to parent
                    # We're adding a zero time apply to make event sequence more useful TODO
                    if ps.parent is not None:
                        ps.next_available_work_time = max(self.ne.current_time, ps.next_available_work_time)
                        self.events.append(PSApplyEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id))
                        self.events.append(SendParamsEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id, ps.parent.id))
                        self.ne.send_msg(ps.id, ps.parent.id, self.msg_size, ps.next_available_work_time)
                        ps.waiting_for_parent = True
                        ps.waiting_child_id = msg.from_id

                    # Otherwise, apply and then send down to child
                    else:
                        apply_start_time = max(self.ne.current_time, ps.next_available_work_time)
                        self.events.append(PSApplyEvent(apply_start_time, apply_start_time + ps.apply_time, ps.id))
                        ps.next_available_work_time = apply_start_time + ps.apply_time
                        self.events.append(SendParamsEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id, msg.from_id))
                        self.ne.send_msg(ps.id, msg.from_id, self.msg_size, ps.next_available_work_time)

                elif ps.sync_style == 'sync':
                    # Only process if all param sets are in

                    ps.n_param_sets_received += 1
                    self.events.append(ReceiveParamsEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id))

                    if ps.n_param_sets_received == len(ps.children):

                        # Add aggr and apply events
                        self.events.append(PSAggrEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                        self.events.append(PSApplyEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id))

                        # If this is a mid level ps, send up to parent
                        if ps.parent is not None:
                            self.events.append(SendParamsEvent(self.ne.current_time + ps.aggr_time + ps.apply_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id, ps.parent.id))
                            self.ne.send_msg(ps.id, ps.parent.id, self.msg_size, self.ne.current_time + ps.aggr_time + ps.apply_time)
                            ps.waiting_for_parent = True

                        # Otherwise, send down to children
                        else:
                            for child in ps.children:
                                self.events.append(SendParamsEvent(self.ne.current_time + ps.aggr_time + ps.apply_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id, child.id))
                                self.ne.send_msg(ps.id, child.id, self.msg_size, self.ne.current_time + ps.aggr_time + ps.apply_time)

                        ps.n_param_sets_received = 0


        elif type(self.nodes[msg.to_id]) == Worker:
            worker = self.nodes[msg.to_id]

            # Add receive and step events
            self.events.append(ReceiveParamsEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id))

            step_time = worker.step_time - worker.st_variation + random.uniform(0, worker.st_variation*2)
            self.events.append(WorkerStepEvent(self.ne.current_time, self.ne.current_time + step_time, worker.id))

            # Send params to parent
            self.events.append(SendParamsEvent(self.ne.current_time + step_time, self.ne.current_time + step_time, worker.id, worker.parent.id))
            self.ne.send_msg(worker.id, worker.parent.id, self.msg_size, self.ne.current_time + step_time)


    def generate(self, eff_start=None, eff_end=None):
        # Move NE until a msg has sent
        sent_msgs = self.ne.move(eff_start, eff_end)
        while len(sent_msgs) == 0:
            sent_msgs = self.ne.move(eff_start, eff_end)

        # Process sent msgs
        for msg in sent_msgs:
            self._process_msg(msg)
            
    def generate_gantt(self, time_stamp):

        # Make node gantt channels
        gantts = {}
        for node in self.nodes:
            gantts[node.id] = []

        # Populate gantt channels
        for event in self.events:
            if type(event) == WorkerStepEvent:
                gantts[event.worker_id].append(event)
            elif type(event) == PSAggrEvent or type(event) == PSApplyEvent:
                gantts[event.ps_id].append(event)
            elif type(event) == ReceiveParamsEvent:
                if type(self.nodes[event.receiver_id]) == Worker or event.sender_id == 0: # TODO need to change this logic for > 2 levels
                    gantts[event.receiver_id].append(event)
                else:
                    gantts[event.sender_id].append(event)

        # Generate row array string
        rows = ""

        for node in self.nodes:

            if type(node) == Worker:
                label = str(node.id) + ' Wk'
            else:
                label = str(node.id) + ' PS'


            times_str = ""
            step_num = 1

            for event in gantts[node.id]:

                block_label_str = ""

                if type(event) == WorkerStepEvent:
                    raw_str = "Working - S: {0}, E: {1}, D: {2}".format(event.start_time, event.end_time, event.end_time - event.start_time)
                    block_label_str = ', "label": {0}'.format(step_num)
                    step_num += 1
                    color = '#5da5c9'
                elif type(event) == PSAggrEvent or type(event) == PSApplyEvent:
                    raw_str = "Updating params - S: {0}, E: {1}, D: {2}".format(event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#003366'
                elif type(event) == ReceiveParamsEvent and event.sender_id == node.id:
                    raw_str = 'Sending to {0} - S: {1}, E: {2}, D: {3}'.format(event.receiver_id, event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#c9c9c9'
                elif type(event) == ReceiveParamsEvent and event.receiver_id == node.id:
                    raw_str = 'Receiving from {0} - S: {1}, E: {2}, D: {3}'.format(event.sender_id, event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#919191'

                times_str += '{{"starting_time": {0}, "ending_time": {1}, "raw": "{2}", "color": "{3}"{4}}},'.format(event.start_time, event.end_time, raw_str, color, block_label_str)

            row = '{{ "label": "{0}", "times": [ {1} ]}},'.format(label, times_str[0:-1])
            rows += row

        row_array_str = "[" + rows[0:-1] + ']'

        # Generate timing breakdown data
        # TODO move this into get_timing_breakdown probably
        timing = self.get_timing_breakdown()

        total_info = {
            'computation': 0,
            'transmission': 0,
            'idle': 0
        }
        total_time = 0

        for node_id in timing:
            node_total_time = 0
            for k in ['computation', 'transmission', 'idle']:
                node_total_time += timing[node_id][k]
                total_time += timing[node_id][k]
                total_info[k] += timing[node_id][k]
            for k in ['computation', 'transmission', 'idle']:
                timing[node_id]['percent_' + k] = timing[node_id][k] / node_total_time

        for k in ['computation', 'transmission', 'idle']:
            total_info['percent_' + k] = total_info[k] / total_time

        timing['total'] = total_info

        timing_str = json.dumps(timing)

        # Generate file
        output = """
        {{
            "timing": {0},
            "rows": {1}
        }}
        """.format(timing_str, row_array_str)

        outfile = open('gantt/gantt_datas/gantt_data_%s.json' % time_stamp, 'w')
        outfile.write(output)
        outfile.close()

    def get_timing_breakdown(self):

        timing = {}
        events_by_node_id = {}
        for node in self.nodes:
            timing[node.id] = {
                'computation': 0,
                'transmission': 0,
                'idle': 0
            }
            events_by_node_id[node.id] = []
        
        for event in self.events:
            if type(event) == WorkerStepEvent:
                events_by_node_id[event.worker_id].append(event)
            elif type(event) == PSAggrEvent or type(event) == PSApplyEvent:
                events_by_node_id[event.ps_id].append(event)
            elif type(event) == ReceiveParamsEvent:
                events_by_node_id[event.sender_id].append(event)

        for node_id in events_by_node_id:
            events_by_node_id[node_id].sort(key=lambda e: e.start_time)

            if len(events_by_node_id[node_id]) != 0:
                timing[node_id]['idle'] += events_by_node_id[node_id][0].start_time
                current_time = events_by_node_id[node_id][0].start_time

            for event in events_by_node_id[node_id]:

                if event.start_time > current_time:
                    timing[node_id]['idle'] += event.start_time - current_time
                    current_time = event.start_time

                if type(event) in [WorkerStepEvent, PSAggrEvent, PSApplyEvent]:
                    if event.start_time < current_time < event.end_time:
                        timing[node_id]['transmission'] -= current_time - event.start_time
                    elif event.start_time < current_time:
                        timing[node_id]['transmission'] -= event.end_time - event.start_time

                    timing[node_id]['computation'] += event.end_time - event.start_time
                    current_time = event.end_time

                elif type(event) == ReceiveParamsEvent:
                    if event.end_time > current_time:
                        timing[node_id]['transmission'] += event.end_time - current_time
                        current_time = event.end_time
                    
        return timing


if __name__ == '__main__':
    import json, sys

    def load_config(config_file_path):
        with open(config_file_path) as config_file:
            config = json.load(config_file)
            config_file.close()
        return config

    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        config_file_path = 'config.json'

    config = load_config(config_file_path)

    nsg = NetworkSequenceGenerator(config['nodes'], 17039680)

    for _ in range(200):
        nsg.generate()

    now = datetime.datetime.now()
    time_str = str(now.time())
    time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')
    nsg.generate_gantt(time_stamp)
