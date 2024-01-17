from NetworkEmulatorLite import NetworkEmulatorLite
from NetworkSequenceGenerator import NetworkSequenceGenerator, UpdateType
from csv_to_configs import make_config
from TwoPassCluster import TwoPassCluster
from madb.vgg16_cifar10 import model_builder, dataset_fn, test_dataset



# Test NetworkEmulator


# 1 lone msg

# A completes during its rampup
def ne_test_1a():
    node_bws = ({ 0: 1, 1: 1 }, { 0: 1, 1: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 0.25, 0)

    ne.move()
    ne.move()

# A completes after its rampup
def ne_test_1b():
    node_bws = ({ 0: 1, 1: 1 }, { 0: 1, 1: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)

    ne.move()
    ne.move()


# 2 msgs come in at the same time

# they complete during their rampup
def ne_test_2a():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 0.1, 0)
    ne.send_msg(0, 2, 0.1, 0)

    for i in range(4):
        ne.move()

# they complete after their rampup
def ne_test_2b():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 1, 0)

    for i in range(4):
        ne.move()


# B comes in during A's rampup

# B comes in before A reaches new DSR

# B completes during A's rampup
def ne_test_3aa():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 0.1, 0.25)

    return ne

# B completes after A's rampup but before A completes
def ne_test_3ab():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 0.75, 0.25)

    return ne

# B completes after A's completes
def ne_test_3ac():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 1, 0.25)

    return ne


# B comes in after A reaches new DSR

# B completes before A
def ne_test_3ba():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 0.1, 0.75)

    return ne

# B completes after A
def ne_test_3bb():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 1, 0.75)

    return ne



# B comes in after A's rampup

# A completes after B
def ne_test_4a():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 0.1, 1.1)

    return ne

# A completes during B's rampup
def ne_test_4b():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 1, 0)
    ne.send_msg(0, 2, 1, 1.25)

    return ne

# A completes after B's rampup
def ne_test_4c():
    node_bws = ({ 0: 1, 1: 1, 2: 1 }, { 0: 1, 1: 1, 2: 1 })
    ne = NetworkEmulatorLite(node_bws, True, False)

    ne.send_msg(0, 1, 2, 0)
    ne.send_msg(0, 2, 1, 1.25)

    return ne


def test_ne(ne_test_fn):
    ne = ne_test_fn()

    sent_msgs = []
    while len(sent_msgs) == 0:
        sent_msgs = ne.move(None)
        for msg in sent_msgs:
            print('from {0} to {1}, start {2}, end {3}'.format(msg.from_id, msg.to_id, msg.start_time, msg.end_time))
        sent_msgs = []




# Test NSG
        
node_descs_1 = [
    {
        'id': 0,
        'parent': None,
        'node_type': 'ps',
        'sync_style': 'sync',
        'aggr_time': 1,
        'apply_time': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 1,
        'parent': 0,
        'node_type': 'ps',
        'sync_style': 'sync',
        'aggr_time': 1,
        'apply_time': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 2,
        'parent': 0,
        'node_type': 'ps',
        'sync_style': 'sync',
        'aggr_time': 1,
        'apply_time': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },

    {
        'id': 3,
        'parent': 1,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 4,
        'parent': 1,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 5,
        'parent': 1,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },

    {
        'id': 6,
        'parent': 2,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 7,
        'parent': 2,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    },
    {
        'id': 8,
        'parent': 2,
        'node_type': 'worker',
        'step_time': 1,
        'st_variation': 0,
        'dropout_chance': 0,
        'inbound_bw': 100,
        'outbound_bw': 100
    }
]


# params S-S no dropout
def nsg_test_pnss():
    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-pnss')

# params S-A no dropout
def nsg_test_pnsa():

    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-pnsa')

# params A-S no dropout
def nsg_test_pnas():

    node_descs_1[0]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-pnas')

# params A-A no dropout
def nsg_test_pnaa():

    node_descs_1[0]['sync_style'] = 'async'
    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-pnaa')


# grads S-S no dropout
def nsg_test_gnss():
    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-gnss')

# grads S-A no dropout
def nsg_test_gnsa():

    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-gnsa')

# grads A-S no dropout
def nsg_test_gnas():

    node_descs_1[0]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-gnas')

# grads A-A no dropout
def nsg_test_gnaa():

    node_descs_1[0]['sync_style'] = 'async'
    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'none', False)

    for i in range(200):
        nsg.generate()

    nsg.generate_gantt('test-gnaa')


# Dropout

    
# params S-S with dropout
def nsg_test_pdss():
    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-pdss')

# params S-A with dropout
def nsg_test_pdsa():

    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-pdsa')

# params A-S with dropout
def nsg_test_pdas():

    node_descs_1[0]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-pdas')

# params A-A with dropout
def nsg_test_pdaa():

    node_descs_1[0]['sync_style'] = 'async'
    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.PARAMS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-pdaa')


# Grads with dropout
    
# grads S-S with dropout
def nsg_test_gdss():
    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-gdss')

# grads S-A with dropout
def nsg_test_gdsa():

    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-gdsa')

# grads A-S with dropout
def nsg_test_gdas():

    node_descs_1[0]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-gdas')

# grads A-A with dropout
def nsg_test_gdaa():

    node_descs_1[0]['sync_style'] = 'async'
    node_descs_1[1]['sync_style'] = 'async'
    node_descs_1[2]['sync_style'] = 'async'

    for node in node_descs_1:
        if node['node_type'] == 'worker':
            node['st_variation'] = 0.5

    for node in node_descs_1:
        if node['node_type'] == 'ps':
            node['apply_time'] = 1

    nsg = NetworkSequenceGenerator(node_descs_1, 100_000_000, True, UpdateType.GRADS, 'nbbl', False)

    for i in range(50):
        nsg.generate()

    nsg.nodes[3].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.nodes[4].should_dropout = True

    for i in range(50):
        nsg.generate()

    nsg.generate_gantt('test-gdaa')




# Test TwoPassCluster process_event
    
raw_config_pss = {'update_type': 'params', 'rb_strat': 'nbbl', 'sync_config': 'S-S', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_psa = {'update_type': 'params', 'rb_strat': 'nbbl', 'sync_config': 'S-A', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_pas = {'update_type': 'params', 'rb_strat': 'nbbl', 'sync_config': 'A-S', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_paa = {'update_type': 'params', 'rb_strat': 'nbbl', 'sync_config': 'A-A', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}

raw_config_gss = {'update_type': 'grads', 'rb_strat': 'nbbl', 'sync_config': 'S-S', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_gsa = {'update_type': 'grads', 'rb_strat': 'nbbl', 'sync_config': 'S-A', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_gas = {'update_type': 'grads', 'rb_strat': 'nbbl', 'sync_config': 'A-S', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}
raw_config_gaa = {'update_type': 'grads', 'rb_strat': 'nbbl', 'sync_config': 'A-A', 'topology': '3-3', 'bw': '100', 'w_step_time': '1', 'w_step_var': '0.5', 'ps_aggr_time': '1', 'ps_apply_time': '1', 'global_dropout_chance': '0', 'epochs': '50', 'target_acc_test': '0.95', 'target_acc_train': '0.5', 'generate_gantt': '1', 'trainless': '0', 'n_runs': '1', 'stop_at_target_test': '0', 'stop_at_target_train': '0', 'bypass_NI': '0', 'network_style': 'hd', 'eval_interval': '782', 'madb_file': '', 'node_config_file': ''}

def test_pe():
    config = make_config(raw_config_gaa)
    tpc = TwoPassCluster(model_builder, dataset_fn, test_dataset, config)

    # tpc.nsg.nodes[3].should_dropout = True

    for i in range(100):
        tpc.nsg.generate()

    tpc.nsg.generate_gantt('test_pe')

    for i in range(12):
        event = tpc.nsg.events.pop(0)
        tpc.process_event(event)

    for i in range(200):
        event = tpc.nsg.events.pop(0)
        tpc.process_event(event)

if __name__ == '__main__':
    test_pe()

    