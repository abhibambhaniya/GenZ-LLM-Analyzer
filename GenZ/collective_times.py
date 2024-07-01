def get_AR_time(data, num_AR_nodes, system):
    """get_AR_time

    Args:
        data (int): Message size(Bytes) per node to complete all reduce.
        num_AR_nodes (int): Number of nodes among which all-reduce is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Reduce
    """

    ## Ring AR Time = Start Latency + N*Tlink +  2M*(N-1)/(N*BW)   
    ## Source:  https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
    AR_time = (5e-6 + 2*(num_AR_nodes-1)*system.interchip_link_latency +  2 * (num_AR_nodes-1) * (data/num_AR_nodes) / system.interchip_mem_bw)*1000

    return AR_time


def get_message_pass_time(data, system):
    """get_message_pass_time

    Args:
        data (int): Message size(Bytes) per node to pass from 1 decide to next.
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to pass the Message from 1 node to next
    """
                      
    msg_pass_time = ((4.2e-6 + (2-1)*system.interchip_link_latency) +  data / system.interchip_mem_bw)*1000
    return msg_pass_time