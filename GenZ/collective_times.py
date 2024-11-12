def get_AR_time(data, numNodes, system):
    """get_AR_time

    Args:
        data (int): Message size(Bytes) per node to complete all reduce.
        num_AR_nodes (int): Number of nodes among which all-reduce is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Reduce
    """
    if data == 0 or numNodes == 1:
        return 0
    ## Ring AR Time = Start Latency + N*Tlink +  2M*(N-1)/(N*BW)
    ## Source:  https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
    allReduceTime = (5e-6 + 2*(numNodes-1)*system.interchip_link_latency +  2 * (numNodes-1) * (data/numNodes) / system.interchip_link_bw)*1000

    return allReduceTime

def get_AG_time(data, numNodes, system):
    """get_AG_time

    Args:
        data (int): Message size(Bytes) per node to complete all gather.
        num_AG_nodes (int): Number of nodes among which all-gather is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Gather
    """
    if data == 0 or numNodes == 1:
        return 0
    ## Ring AG Time = Start Latency + N*Tlink +  2M*(N-1)/(N*BW)
    ## Source:  https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
    allGatherDuration = (5e-6 + (numNodes-1)*system.interchip_link_latency +  (numNodes-1) * (data/numNodes) / system.interchip_link_bw)*1000

    return allGatherDuration

def get_message_pass_time(data, system):
    """get_message_pass_time

    Args:
        data (int): Message size(Bytes) per node to pass from 1 decide to next.
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to pass the Message from 1 node to next
    """
    if data == 0:
        return 0
    msg_pass_time = (system.interchip_link_latency +  data / system.interchip_link_bw)*1000
    return msg_pass_time


def get_A2A_time(data, numNodes, system):
    """get_A2A_time

    Args:
        data (int): Total message size (Bytes) per node to be exchanged in all-to-all.
        num_A2A_nodes (int): Number of nodes participating in the all-to-all operation.
        system (System object): Object of class System

    Returns:
        time (float): Total time (msec) to complete the All-to-All operation
    """

    ## BWeff = 4B/N if Ring of size N
    ## BWeff = 4B/T if 2D Torus of size TxT

    # M = E/ep * D/tp * F * B * bb

    # A2A time = Start Latency + (N-1) * Tlink + (N-1) * M / BW
    # Where N is the number of nodes, M is the message size per node pair,
    # Tlink is the inter-chip link latency, and BW is the inter-chip memory bandwidth

    if data == 0 or numNodes == 1:
        return 0
    message_size_per_pair = data / numNodes
    A2A_time = (5e-6 + (numNodes - 1) * system.interchip_link_latency +
                (numNodes - 1) * message_size_per_pair / system.interchip_link_bw) * 1000

    return A2A_time