from NetworkEmulatorLite import NetworkEmulatorLite



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


if __name__ == '__main__':
    ne = ne_test_4c()

    sent_msgs = []
    while len(sent_msgs) == 0:
        sent_msgs = ne.move(None)
        for msg in sent_msgs:
            print('from {0} to {1}, start {2}, end {3}'.format(msg.from_id, msg.to_id, msg.start_time, msg.end_time))
        sent_msgs = []