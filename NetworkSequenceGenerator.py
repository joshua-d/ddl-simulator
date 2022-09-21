import datetime
from NetworkEmulatorLite import NetworkEmulatorLite


# TODO this
PARAMS_SIZE = 100


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


event_types = [
    WorkerStepEvent,
    SendParamsEvent,
    ReceiveParamsEvent,
    PSAggrEvent,
    PSApplyEvent
]


gantt_color_map = {
    WorkerStepEvent: '#5da5c9',
    PSAggrEvent: '#003366',
    PSApplyEvent: '#003366',
    SendParamsEvent: '#c9c9c9',
    ReceiveParamsEvent: '#919191'
}


class Worker:
    def __init__(self, id, step_time):
        self.id = id
        self.parent = None

        self.step_time = step_time


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


class NetworkSequenceGenerator:

    def __init__(self, node_descs):

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
                inbound_max[ps.id] = node_desc['inbound_bw']
                outbound_max[ps.id] = node_desc['outbound_bw']

            elif node_desc['node_type'] == 'worker':
                w = Worker(
                        node_desc['id'],
                        node_desc['step_time']
                    )
                self.nodes.append(w)
                self.workers.append(w)
                inbound_max[w.id] = node_desc['inbound_bw']
                outbound_max[w.id] = node_desc['outbound_bw']

        # Nodes in config must be in order of ID
        # Build parents and children lists
        for node_desc in node_descs:
            node = self.nodes[node_desc['id']]
            if node_desc['parent'] is not None:
                node.parent = self.nodes[node_desc['parent']]
                node.parent.children.append(node)

        # Build NE
        self.ne = NetworkEmulatorLite((inbound_max, outbound_max))

        # Set up starting events
        for worker in self.workers:
            self.events.append(WorkerStepEvent(0, worker.step_time, worker.id))
            self.events.append(SendParamsEvent(worker.step_time, worker.step_time, worker.id, worker.parent.id))
            self.ne.send_msg(worker.id, worker.parent.id, PARAMS_SIZE, worker.step_time, None) # TODO msg id

        

    def _process_msg(self, msg, queued_child_msg=False):

        if type(self.nodes[msg.to_id]) == ParameterServer:

            ps = self.nodes[msg.to_id]

            if ps.waiting_for_parent:
                if msg.from_id == ps.parent.id:
                    # Handle update from parent

                    self.events.append(ReceiveParamsEvent(self.ne.current_time, self.ne.current_time, msg.from_id, msg.to_id))

                    # Immediately send to child(ren)
                    if ps.sync_style == 'async':
                        self.events.append(SendParamsEvent(self.ne.current_time, self.ne.current_time, ps.id, ps.waiting_child_id))
                        self.ne.send_msg(ps.id, ps.waiting_child_id, PARAMS_SIZE, self.ne.current_time, None)
                    elif ps.sync_style == 'sync':
                        for child in ps.children:
                            self.events.append(SendParamsEvent(self.ne.current_time, self.ne.current_time, ps.id, child.id))
                            self.ne.send_msg(ps.id, child.id, PARAMS_SIZE, self.ne.current_time, None)

                    ps.waiting_for_parent = False

                    # Process msgs from children
                    for child_msg in ps.child_msg_queue:
                        self._process_msg(child_msg, queued_child_msg=True)
                    
                    ps.child_msg_queue = []
                    
                else:
                    # Got update from child while waiting for parent
                    self.events.append(ReceiveParamsEvent(self.ne.current_time, self.ne.current_time, msg.from_id, msg.to_id))
                    ps.child_msg_queue.append(msg)

            else:
                # Not waiting for parent, update is from child

                if ps.sync_style == 'async':
                    # Process params immediately

                    # Add receive and apply events
                    if not queued_child_msg:
                        self.events.append(ReceiveParamsEvent(self.ne.current_time, self.ne.current_time, msg.from_id, msg.to_id))
                    self.events.append(PSApplyEvent(self.ne.current_time, self.ne.current_time + ps.apply_time, ps.id))

                    # If this is a mid level ps, send up to parent
                    if ps.parent is not None:
                        self.events.append(SendParamsEvent(self.ne.current_time + ps.apply_time, self.ne.current_time + ps.apply_time, ps.id, ps.parent.id))
                        self.ne.send_msg(ps.id, ps.parent.id, PARAMS_SIZE, self.ne.current_time + ps.apply_time, None)
                        ps.waiting_for_parent = True
                        ps.waiting_child_id = msg.from_id

                    # Otherwise, send down to child
                    else:
                        self.events.append(SendParamsEvent(self.ne.current_time + ps.apply_time, self.ne.current_time + ps.apply_time, ps.id, msg.from_id))
                        self.ne.send_msg(ps.id, msg.from_id, PARAMS_SIZE, self.ne.current_time + ps.apply_time, None)

                elif ps.sync_style == 'sync':
                    # Only process if all param sets are in

                    ps.n_param_sets_received += 1
                    
                    if not queued_child_msg:
                        self.events.append(ReceiveParamsEvent(self.ne.current_time, self.ne.current_time, msg.from_id, msg.to_id))

                    if ps.n_param_sets_received == len(ps.children):

                        # Add aggr and apply events
                        self.events.append(PSAggrEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                        self.events.append(PSApplyEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id))

                        # If this is a mid level ps, send up to parent
                        if ps.parent is not None:
                            self.events.append(SendParamsEvent(self.ne.current_time + ps.aggr_time + ps.apply_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id, ps.parent.id))
                            self.ne.send_msg(ps.id, ps.parent.id, PARAMS_SIZE, self.ne.current_time + ps.aggr_time + ps.apply_time, None)
                            ps.waiting_for_parent = True

                        # Otherwise, send down to children
                        else:
                            for child in ps.children:
                                self.events.append(SendParamsEvent(self.ne.current_time + ps.aggr_time + ps.apply_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id, child.id))
                                self.ne.send_msg(ps.id, child.id, PARAMS_SIZE, self.ne.current_time + ps.aggr_time + ps.apply_time, None)

                        ps.n_param_sets_received = 0


        elif type(self.nodes[msg.to_id]) == Worker:
            worker = self.nodes[msg.to_id]

            # Add receive and step events
            self.events.append(ReceiveParamsEvent(self.ne.current_time, self.ne.current_time, msg.from_id, msg.to_id))
            self.events.append(WorkerStepEvent(self.ne.current_time, self.ne.current_time + worker.step_time, worker.id))

            # Send params to parent
            self.events.append(SendParamsEvent(self.ne.current_time + worker.step_time, self.ne.current_time + worker.step_time, worker.id, worker.parent.id))
            self.ne.send_msg(worker.id, worker.parent.id, PARAMS_SIZE, self.ne.current_time + worker.step_time, None)


    def generate(self):
        # Move NE until a msg has sent
        sent_msgs = self.ne.move()
        while len(sent_msgs) == 0:
            sent_msgs = self.ne.move()

        # Process sent msgs
        for msg in sent_msgs:
            self._process_msg(msg)
            
    def generate_gantt(self):
        # This can only be run once generation is done <3

        # Set send/receive start times
        event_idx = 0
        while event_idx < len(self.events):
            event = self.events[event_idx]
            if type(event) == SendParamsEvent:
                comp_event_idx = event_idx + 1
                while comp_event_idx < len(self.events):
                    comp_event = self.events[comp_event_idx]
                    if type(comp_event) == ReceiveParamsEvent and comp_event.sender_id == event.sender_id and comp_event.receiver_id == event.receiver_id:
                        event.end_time = comp_event.end_time
                        comp_event.start_time = event.start_time
                        break
                    comp_event_idx += 1
            event_idx += 1

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
            elif type(event) == SendParamsEvent:
                if type(self.nodes[event.receiver_id]) == Worker or event.sender_id == 0: # TODO need to change this logic for > 2 levels
                    gantts[event.receiver_id].append(event)
                else:
                    gantts[event.sender_id].append(event)

        # Get time stamp
        now = datetime.datetime.now()
        time_str = str(now.time())
        time_stamp = str(now.date()) + '_' + time_str[0:time_str.find('.')].replace(':', '-')

        # Generate file
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
                elif type(event) == SendParamsEvent and event.sender_id == node.id:
                    raw_str = 'Sending to {0} - S: {1}, E: {2}, D: {3}'.format(event.receiver_id, event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#c9c9c9'
                elif type(event) == SendParamsEvent and event.receiver_id == node.id:
                    raw_str = 'Receiving from {0} - S: {1}, E: {2}, D: {3}'.format(event.sender_id, event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#919191'

                times_str += '{{"starting_time": {0}, "ending_time": {1}, "raw": "{2}", "color": "{3}"{4}}},'.format(event.start_time, event.end_time, raw_str, color, block_label_str)

            row = '{{ "label": "{0}", "times": [ {1} ]}},'.format(label, times_str[0:-1])
            rows += row

        res = "[" + rows[0:-1] + ']'

        outfile = open('gantt/gantt_datas/gantt_data_%s.json' % time_stamp, 'w')
        outfile.write(res)
        outfile.close()


import json
node_desc_str = """
[
    {
        "node_type": "ps",
        "id": 0,
        "parent": null,
        "sync_style": "async",
        "aggr_time": 1,
        "apply_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },
    {
        "node_type": "ps",
        "id": 1,
        "parent": 0,
        "sync_style": "async",
        "aggr_time": 1,
        "apply_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },
    {
        "node_type": "ps",
        "id": 2,
        "parent": 0,
        "sync_style": "async",
        "aggr_time": 1,
        "apply_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },

    {
        "node_type": "worker",
        "id": 3,
        "parent": 1,
        "step_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },
    {
        "node_type": "worker",
        "id": 4,
        "parent": 1,
        "step_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },
    {
        "node_type": "worker",
        "id": 5,
        "parent": 2,
        "step_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    },
    {
        "node_type": "worker",
        "id": 6,
        "parent": 2,
        "step_time": 1,

        "inbound_bw": 100,
        "outbound_bw": 100
    }
]
"""

node_descs = json.loads(node_desc_str)

nsg = NetworkSequenceGenerator(node_descs)

for _ in range(20):
    nsg.generate()

nsg.generate_gantt()