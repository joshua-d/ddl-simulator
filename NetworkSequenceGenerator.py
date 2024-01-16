import datetime, random
from NetworkEmulatorLite import NetworkEmulatorLite
import json
from enum import Enum
from math import inf


class RebalancingStrategy(Enum):
    SBBL = 0  # These numbers are important because they are used as indices in worker score array
    BWBBL = 1
    OABBL = 2
    NBBL = 3


RB_Strategy = None


class UpdateType(Enum):
    PARAMS = 0
    GRADS = 1


class Event:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


class WorkerStepEvent(Event):
    def __init__(self, start_time, end_time, worker_id):
        super().__init__(start_time, end_time)
        self.worker_id = worker_id


class SendUpdateEvent(Event):
    def __init__(self, start_time, end_time, sender_id, receiver_id):
        super().__init__(start_time, end_time)
        self.sender_id = sender_id
        self.receiver_id = receiver_id


class ReceiveUpdateEvent(Event):
    def __init__(self, start_time, end_time, sender_id, receiver_id, update_type):
        super().__init__(start_time, end_time)
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.update_type = update_type

        
class PSAggrParamsEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSSaveParamsEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSSaveParamsFromParentEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSAggrGradsEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSApplyGradsEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class PSSaveGradsEvent(Event):
    def __init__(self, start_time, end_time, ps_id):
        super().__init__(start_time, end_time)
        self.ps_id = ps_id


class DropoutEvent(Event):
    def __init__(self, start_time, end_time, worker_id, parent_id, breakdown):
        super().__init__(start_time, end_time)
        self.worker_id = worker_id
        self.parent_id = parent_id
        self.breakdown = breakdown


class RebalanceEvent(Event):
    def __init__(self, start_time, end_time, worker_id, old_parent_id, new_parent_id, breakdown):
        super().__init__(start_time, end_time)
        self.worker_id = worker_id
        self.old_parent_id = old_parent_id
        self.new_parent_id = new_parent_id
        self.breakdown = breakdown




class Worker:
    def __init__(self, id, step_time, st_variation, dropout_chance):
        self.id = id
        self.parent = None

        self.step_time = step_time
        self.st_variation = st_variation
        self.dropout_chance = dropout_chance

        self.dropped_out = False
        self.score = [0, 0, 0]


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

    def __init__(self, node_descs, msg_size, half_duplex, update_type, rb_strat, bypass_NI):

        # In bits
        self.msg_size = msg_size
        self.update_type = update_type

        self.nodes = []
        self.workers = []
        self.parameter_servers = []

        self.events = []

        inbound_max = {}
        outbound_max = {}

        self.n_batches = 0

        global RB_Strategy
        if rb_strat == 'nbbl':
            RB_Strategy = RebalancingStrategy.NBBL
        elif rb_strat == 'oabbl':
            RB_Strategy = RebalancingStrategy.OABBL
        elif rb_strat == 'sbbl':
            RB_Strategy = RebalancingStrategy.SBBL
        elif rb_strat == 'bwbbl':
            RB_Strategy = RebalancingStrategy.BWBBL

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
                        node_desc['st_variation'],
                        node_desc['dropout_chance']
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
        self.ne = NetworkEmulatorLite((inbound_max, outbound_max), half_duplex, bypass_NI)

        # Calculate processing power scores for workers
        max_inbound_bw = 0
        max_outbound_bw = 0
        min_step_time = -1
        for worker in self.workers:
            if inbound_max[worker.id] > max_inbound_bw:
                max_inbound_bw = inbound_max[worker.id]
            if outbound_max[worker.id] > max_outbound_bw:
                max_outbound_bw = outbound_max[worker.id]
            if worker.step_time < min_step_time or min_step_time == -1:
                min_step_time = worker.step_time

        for worker in self.workers:
            # Score formulas here TODO
            # Speed score
            worker.score[0] = min_step_time / worker.step_time
            # BW Score
            worker.score[1] = inbound_max[worker.id] / max_inbound_bw + outbound_max[worker.id] / max_outbound_bw
            # OA Score
            worker.score[2] = worker.score[0] + worker.score[1]

        # Set up starting events
        for worker in self.workers:
            step_time = worker.step_time - worker.st_variation + random.uniform(0, worker.st_variation*2)
            self.events.append(WorkerStepEvent(0, step_time, worker.id))
            self.n_batches += 1
            self.events.append(SendUpdateEvent(step_time, step_time, worker.id, worker.parent.id))
            self.ne.send_msg(worker.id, worker.parent.id, self.msg_size, step_time, self.update_type) # send update type as msg metadata


    # Checks if the system should be rebalanced, then performs the rebalance if it should. Returns (bool - whether or not rebalance occurred, moved worker)
    def _score_rebalance(self, msg, score_idx):
        rebalanced = False

        # Calculate summed proc pow score for each cluster
        scores = {}
        highest_score = -1
        lowest_score = -1
        for worker in self.workers:
            if worker.parent.id not in scores:
                score = 0
                for child in worker.parent.children:
                    score += child.score[score_idx]
                scores[worker.parent.id] = score
                if score > highest_score or highest_score == -1:
                    highest_score = score
                    h_ps = worker.parent
                if score < lowest_score or lowest_score == -1:
                    lowest_score = score
                    l_ps = worker.parent

        if highest_score != lowest_score:

            threshold = (highest_score - lowest_score) / 2

            # Get h_ps worker whose score is closest to threshold
            closest_val = -1
            for child in h_ps.children:
                v = (child.score[score_idx] - threshold)*(child.score[score_idx] - threshold)
                if v < closest_val or closest_val == -1:
                    closest_val = v
                    closest_worker = child

            # If the move would be beneficial, do it
            old_diff = highest_score - lowest_score
            new_diff = (highest_score - closest_worker.score[score_idx]) - (lowest_score + closest_worker.score[score_idx])
            if new_diff*new_diff < old_diff*old_diff:
                # Transfer 1 worker from mc to lc
                h_ps.children[-1].parent = l_ps
                l_ps.children.append(h_ps.children.pop())
                rebalanced = True
                print(f'REBALANCE worker {l_ps.children[-1].id}, ps {h_ps.id} -> {l_ps.id}')
                print(f'h: {highest_score} l: {lowest_score} w: {closest_worker.score[score_idx]}')
                self.events.append(RebalanceEvent(msg.end_time, msg.end_time, l_ps.children[-1].id, h_ps.id, l_ps.id, self.get_topology_breakdown()))

        return rebalanced, l_ps.children[-1]
        

    def _process_msg(self, msg):

        if type(self.nodes[msg.to_id]) == ParameterServer:

            ps = self.nodes[msg.to_id]

            # Disregard msg if it is from Worker who is not a child
            if type(self.nodes[msg.from_id]) == Worker and self.nodes[msg.from_id] not in ps.children:
                return 

            if ps.waiting_for_parent:
                if msg.from_id == ps.parent.id:
                    # Handle update from parent

                    self.events.append(ReceiveUpdateEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id, msg.metadata))

                    # Params are saved immediately on ReceiveUpdateEvent from parent
                    # self.events.append(PSSaveParamsFromParentEvent(msg.end_time, msg.end_time, ps.id))

                    # Immediately send to child(ren)
                    if ps.sync_style == 'async':
                        self.events.append(SendUpdateEvent(self.ne.current_time, self.ne.current_time, ps.id, ps.waiting_child_id))
                        self.ne.send_msg(ps.id, ps.waiting_child_id, self.msg_size, self.ne.current_time, UpdateType.PARAMS)
                    elif ps.sync_style == 'sync':
                        for child in ps.children:
                            self.events.append(SendUpdateEvent(self.ne.current_time, self.ne.current_time, ps.id, child.id))
                            self.ne.send_msg(ps.id, child.id, self.msg_size, self.ne.current_time, UpdateType.PARAMS)

                    ps.waiting_for_parent = False

                    # Process msgs from children
                    if ps.sync_style == 'sync': # TODO sync PS should actually never receive an update from a child while waiting for its parent
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
                    self.events.append(ReceiveUpdateEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id, msg.metadata))
                    
                    # If this is a mid level ps, send up to parent
                    if ps.parent is not None:
                        ps.next_available_work_time = max(self.ne.current_time, ps.next_available_work_time)

                        # Zero-time save event
                        if msg.metadata == UpdateType.PARAMS:
                            self.events.append(PSSaveParamsEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id))
                        else:
                            self.events.append(PSSaveGradsEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id))

                        self.events.append(SendUpdateEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id, ps.parent.id))
                        self.ne.send_msg(ps.id, ps.parent.id, self.msg_size, ps.next_available_work_time, msg.metadata) # pass msg type up
                        ps.waiting_for_parent = True
                        ps.waiting_child_id = msg.from_id

                    # Otherwise, apply and then send down to child
                    else:
                        start_time = max(self.ne.current_time, ps.next_available_work_time)

                        if msg.metadata == UpdateType.PARAMS:
                            self.events.append(PSAggrParamsEvent(start_time, start_time + ps.aggr_time, ps.id))
                            ps.next_available_work_time = start_time + ps.aggr_time
                        else:
                            self.events.append(PSApplyGradsEvent(start_time, start_time + ps.apply_time, ps.id))
                            ps.next_available_work_time = start_time + ps.apply_time

                        self.events.append(SendUpdateEvent(ps.next_available_work_time, ps.next_available_work_time, ps.id, msg.from_id))
                        self.ne.send_msg(ps.id, msg.from_id, self.msg_size, ps.next_available_work_time, UpdateType.PARAMS)

                elif ps.sync_style == 'sync':
                    # Only process if all param sets are in

                    ps.n_param_sets_received += 1
                    self.events.append(ReceiveUpdateEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id, msg.metadata))

                    if ps.n_param_sets_received >= len(ps.children):

                        # If this is a mid level ps, send up to parent
                        if ps.parent is not None:

                            # Aggregate (but don't apply grads)
                            if msg.metadata == UpdateType.PARAMS:
                                self.events.append(PSAggrParamsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                            else:
                                self.events.append(PSAggrGradsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))

                            self.events.append(SendUpdateEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time, ps.id, ps.parent.id))
                            self.ne.send_msg(ps.id, ps.parent.id, self.msg_size, self.ne.current_time + ps.aggr_time, msg.metadata)
                            ps.waiting_for_parent = True

                        # Otherwise, send down to children
                        else:

                            # Aggregate and apply
                            if msg.metadata == UpdateType.PARAMS:
                                self.events.append(PSAggrParamsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                                send_update_start_time = self.ne.current_time + ps.aggr_time
                            else:
                                self.events.append(PSAggrGradsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                                self.events.append(PSApplyGradsEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id))
                                send_update_start_time = self.ne.current_time + ps.aggr_time + ps.apply_time

                            for child in ps.children:
                                self.events.append(SendUpdateEvent(send_update_start_time, send_update_start_time, ps.id, child.id))
                                self.ne.send_msg(ps.id, child.id, self.msg_size, send_update_start_time, UpdateType.PARAMS)

                        ps.n_param_sets_received = 0


        elif type(self.nodes[msg.to_id]) == Worker:
            worker = self.nodes[msg.to_id]

            # If not from my parent, don't read
            if msg.from_id != worker.parent.id:
                return

            # Add receive and step events
            self.events.append(ReceiveUpdateEvent(msg.start_time, msg.end_time, msg.from_id, msg.to_id, msg.metadata))

            # Check for dropout
            if random.uniform(0, 1) < worker.dropout_chance and len(worker.parent.children) > 1: # TODO second case is just for now - last worker in a cluster can't drop out
                
                print(f'DROPOUT: worker {worker.id}, parent {worker.parent.id}')
                ps = worker.parent
                ps.children.remove(worker)
                worker.dropped_out = True
                self.events.append(DropoutEvent(msg.end_time, msg.end_time, worker.id, worker.parent.id, self.get_topology_breakdown()))

                rebalanced = False

                # Rebalance
                # TODO - assumes balanced at beginning - moves 1 worker per dropout
                # assumes cluster that a worker gets moved to is the one that was just dropped out of (worker.parent)
                if RB_Strategy == RebalancingStrategy.NBBL:
                    # Load relevant PSs
                    least_children = -1
                    most_children = -1

                    for worker in self.workers:
                        if len(worker.parent.children) < least_children or least_children == -1:
                            least_children = len(worker.parent.children)
                            l_ps = worker.parent
                        if len(worker.parent.children) > most_children or most_children == -1:
                            most_children = len(worker.parent.children)
                            h_ps = worker.parent

                    if most_children - least_children > 1:
                        # Transfer 1 worker from h to l
                        h_ps.children[-1].parent = l_ps
                        l_ps.children.append(h_ps.children.pop())
                        rebalanced = True
                        moved_worker = l_ps.children[-1]
                        print(f'REBALANCE worker {l_ps.children[-1].id}, ps {h_ps.id} -> {l_ps.id}')
                        self.events.append(RebalanceEvent(msg.end_time, msg.end_time, l_ps.children[-1].id, h_ps.id, l_ps.id, self.get_topology_breakdown()))

                elif RB_Strategy == RebalancingStrategy.SBBL or RB_Strategy == RebalancingStrategy.BWBBL or RB_Strategy == RebalancingStrategy.OABBL:
                    rebalanced, moved_worker = self._score_rebalance(msg, RB_Strategy.value)

                # Check each sync PS for sync round completion
                for ps in self.parameter_servers:
                    if ps.sync_style == 'sync':
                        if ps.n_param_sets_received >= len(ps.children): # TODO this is the same code copied from above

                            # If this is a mid level ps, send up to parent
                            if ps.parent is not None:

                                # Aggregate (but don't apply grads)
                                if msg.metadata == UpdateType.PARAMS:
                                    self.events.append(PSAggrParamsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                                else:
                                    self.events.append(PSAggrGradsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))

                                self.events.append(SendUpdateEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time, ps.id, ps.parent.id))
                                self.ne.send_msg(ps.id, ps.parent.id, self.msg_size, self.ne.current_time + ps.aggr_time, msg.metadata)
                                ps.waiting_for_parent = True

                            # Otherwise, send down to children
                            else:

                                # Aggregate and apply
                                if msg.metadata == UpdateType.PARAMS:
                                    self.events.append(PSAggrParamsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                                    send_update_start_time = self.ne.current_time + ps.aggr_time
                                else:
                                    self.events.append(PSAggrGradsEvent(self.ne.current_time, self.ne.current_time + ps.aggr_time, ps.id))
                                    self.events.append(PSApplyGradsEvent(self.ne.current_time + ps.aggr_time, self.ne.current_time + ps.aggr_time + ps.apply_time, ps.id))
                                    send_update_start_time = self.ne.current_time + ps.aggr_time + ps.apply_time

                                for child in ps.children:
                                    self.events.append(SendUpdateEvent(send_update_start_time, send_update_start_time, ps.id, child.id))
                                    self.ne.send_msg(ps.id, child.id, self.msg_size, send_update_start_time, UpdateType.PARAMS)

                            ps.n_param_sets_received = 0

                # Moved worker does step
                if rebalanced:
                    worker = moved_worker
                    step_time = worker.step_time - worker.st_variation + random.uniform(0, worker.st_variation*2)
                    self.events.append(WorkerStepEvent(self.ne.current_time, self.ne.current_time + step_time, worker.id))
                    self.n_batches += 1
                    self.events.append(SendUpdateEvent(self.ne.current_time + step_time, self.ne.current_time + step_time, worker.id, worker.parent.id))
                    self.ne.send_msg(worker.id, worker.parent.id, self.msg_size, self.ne.current_time + step_time, self.update_type)

            else:
                # Send params to parent
                step_time = worker.step_time - worker.st_variation + random.uniform(0, worker.st_variation*2)
                self.events.append(WorkerStepEvent(self.ne.current_time, self.ne.current_time + step_time, worker.id))
                self.n_batches += 1
                self.events.append(SendUpdateEvent(self.ne.current_time + step_time, self.ne.current_time + step_time, worker.id, worker.parent.id))
                self.ne.send_msg(worker.id, worker.parent.id, self.msg_size, self.ne.current_time + step_time, self.update_type)


    def generate(self, end_time=None, end_batch=None, eff_start=None, eff_end=None):
        # Move NE until a msg has sent
        sent_msgs = self.ne.move(eff_start, eff_end)

        while len(sent_msgs) == 0:
            if (end_time is not None and self.ne.current_time >= end_time) or (end_batch is not None and self.n_batches >= end_batch):
                return True
            sent_msgs = self.ne.move(eff_start, eff_end)
            
        # Process sent msgs
        for msg in sent_msgs:
            self._process_msg(msg)

        if (end_time is not None and self.ne.current_time >= end_time) or (end_batch is not None and self.n_batches >= end_batch):
            return True

        return False
            
    def generate_gantt(self, stamp):

        # Make node gantt channels
        gantts = {}
        for node in self.nodes:
            gantts[node.id] = []

        # Populate gantt channels
        for event in self.events:
            if type(event) == WorkerStepEvent:
                gantts[event.worker_id].append(event)
            elif type(event) == PSAggrParamsEvent or type(event) == PSSaveParamsEvent:
                gantts[event.ps_id].append(event)
            elif type(event) == ReceiveUpdateEvent:
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
                elif type(event) == PSAggrParamsEvent or type(event) == PSSaveParamsEvent:
                    raw_str = "Updating params - S: {0}, E: {1}, D: {2}".format(event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#003366'
                elif type(event) == ReceiveUpdateEvent and event.sender_id == node.id:
                    raw_str = 'Sending to {0} - S: {1}, E: {2}, D: {3}'.format(event.receiver_id, event.start_time, event.end_time, event.end_time - event.start_time)
                    color = '#c9c9c9'
                elif type(event) == ReceiveUpdateEvent and event.receiver_id == node.id:
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

        outfile = open('gantt/gantt_datas/gantt_data_%s.json' % stamp, 'w')
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
            elif type(event) == PSAggrParamsEvent or type(event) == PSSaveParamsEvent:
                events_by_node_id[event.ps_id].append(event)
            elif type(event) == ReceiveUpdateEvent:
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

                if type(event) in [WorkerStepEvent, PSAggrParamsEvent, PSSaveParamsEvent]:
                    if event.start_time < current_time < event.end_time:
                        timing[node_id]['transmission'] -= current_time - event.start_time
                    elif event.start_time < current_time:
                        timing[node_id]['transmission'] -= event.end_time - event.start_time

                    timing[node_id]['computation'] += event.end_time - event.start_time
                    current_time = event.end_time

                elif type(event) == ReceiveUpdateEvent:
                    if event.end_time > current_time:
                        timing[node_id]['transmission'] += event.end_time - current_time
                        current_time = event.end_time
                    
        return timing

    def get_topology_breakdown(self):
        # res = ''
        # for ps in self.parameter_servers:
        #     score = 0
        #     for child in ps.children:
        #         if type(child) != Worker:
        #             break
        #         score += child.oa_score

        #     res += f'PS ID: {ps.id}, \tSc: {score}, \tN: {len(ps.children)}, \tC: '

        #     for child in ps.children:
        #         res += f'{child.id}, '
        #     res = res[0:-2] + '\n'

        # return res

        res = ''
        for ps in self.parameter_servers:
            score = 0
            for child in ps.children:
                if type(child) != Worker:
                    break
                score += child.score[2]

            res += f'P{ps.id}: {len(ps.children)}, {score}\t'

        return res



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
