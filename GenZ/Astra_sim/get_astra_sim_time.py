
import warnings
warnings.filterwarnings("ignore", message="UserWarning: Protobuf gencode version 5.27.2 is older than")
import os
import subprocess
import yaml
from GenZ.system import System
from GenZ.unit import Unit
import numpy as np
import contextlib
import io
from .fix_chakra_traces import convert_chakra_file
import re

txt_file_path = "/tmp/genz/chakra/txt_file.txt"
et_output_path = "/tmp/genz/chakra/et/"
et_cleaned_output_path = "/tmp/genz/chakra/et_cleaned"

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
run_file = os.path.join(SCRIPT_DIR, "run.sh")
ASTRA_SIM_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "astra_output.txt") 

def merge_parallelism_heirarchy(parallelism_heirarchy:str, merge_dim='EP', merge_into='TP') -> str:
    """
    parallelism_heirarchy: str: A string with the following format: Ex: "TP{x}_EP{y}_PP{z}"
    merge_dim: str: A string with the following format: Ex: "EP"
    merge_into: str: A string with the following format: Ex: "TP"
    
    The function merges the parallelism dimension 'merge_dim' into 'merge_into' dimension.
    # Ex1:parallelism_heirarchy = 'TP{2}_EP{4}_PP{2}' , merge_dim = EP, merge_into = TP
    # output = 'TP{8}_PP{2}'
    # Ex2:parallelism_heirarchy = 'PP{2}_EP{4}_TP{2}' , merge_dim = EP, merge_into = TP
    # output = 'PP{2}_TP{8}'
    
    return: str: A string with the following format: Ex: "TP{x}_EP{y}_PP{z}"
    """
    
    pattern = r'\{(\d+)\}'
    parallelism_sizes = re.findall(pattern, parallelism_heirarchy)   
    merge_final_position = re.sub(pattern,'',parallelism_heirarchy ).split('_').index(merge_into)
    to_merge_dim_position = re.sub(pattern,'',parallelism_heirarchy ).split('_').index(merge_dim)
    
    merge_final_size = int(parallelism_sizes[merge_final_position]) * int(parallelism_sizes[to_merge_dim_position])
    
    if to_merge_dim_position == 0:
        new_parallelism_heirarchy = parallelism_heirarchy[parallelism_heirarchy.index('_') + 1:]
        new_parallelism_heirarchy = re.sub(f'{merge_into}\{{\d+\}}', f'{merge_into}{{{merge_final_size}}}', new_parallelism_heirarchy)
    else:
        new_parallelism_heirarchy = re.sub(f'_{merge_dim}\{{\d+\}}','',parallelism_heirarchy ) 
        new_parallelism_heirarchy = re.sub(f'{merge_into}\{{\d+\}}', f'{merge_into}{{{merge_final_size}}}', new_parallelism_heirarchy)
    
    return new_parallelism_heirarchy


def divide_npus_count(network_config, parallelism_sizes):
    result = []
    dims = []
    npus_count = network_config["npus_count"].copy()
    dim_index = np.arange(len(npus_count)).tolist()
    for size in parallelism_sizes:
        current_nodes = []
        current_dims = []
        while np.prod(current_nodes) < size:
            last_dim_nodes = npus_count.pop(0)
            index = dim_index.pop(0)
            if last_dim_nodes * max(1, np.prod(current_nodes)) <= size:
                current_nodes.append(last_dim_nodes)
                current_dims.append(index)
            else:
                dim_used = size/max(1, np.prod(current_nodes))
                current_nodes.append(dim_used)
                current_dims.append(index)
                last_dim_nodes /= dim_used
                npus_count.insert(0, last_dim_nodes)
                dim_index.insert(0, index)
        result.append(current_nodes)
        dims.append(current_dims)
    return result, dims

def get_network_config(network_config:dict, parallelism_heirarchy:str, parallelism:str) -> dict:
    """
    network_config: dict: A dictionary with the following keys:
            "topology":   ## List of topology (“Ring”, “FullyConnected”, or “Switch”)
            "npus_count": ## List of number of npus per node
            "bandwidth":  ## List of Link Bw in GB/s
            "latency":    ## Link expects latency in ns
    parallelism_heirarchy: str: A string with the following format: Ex: "TP{x}_EP{y}_PP{z}"
        In the above example, TP is among the closest nodes, then EP and finally PP accross the last dimension.
    """ 
    assert type(network_config) == dict, "network_config must be a dictionary"
    
    pattern = r'\{(\d+)\}'
    parallelism_sizes = re.findall(pattern, parallelism_heirarchy)
    assert np.prod(network_config["npus_count"]) == np.prod([int(match) for match in parallelism_sizes]), f"Prof of npus_count:{np.prod(network_config['npus_count'])} should be equal to num_nodes:{np.prod([int(match) for match in parallelism_sizes])}"

    assert parallelism in parallelism_heirarchy, "parallelism should be present in parallelism_heirarchy"

    # find the parallelism in parallelism_heirarchy, it would be distributed by '_', find the position. 
    # Ex1:parallelism_heirarchy = 'TP{2}_EP{4}_PP{2}' , parallelism = EP
    # output = 1
    # Ex2:parallelism_heirarchy = 'TP{2}_EP{4}_PP{2}' , parallelism = TP
    # output = 0
    parallelism_position = re.sub(pattern,'',parallelism_heirarchy ).split('_').index(parallelism)
    dims, indexes = divide_npus_count(network_config, [int(match) for match in parallelism_sizes])

    parallelism_index = indexes[parallelism_position]
    network_config_for_parallelism = {}
    for key, value in network_config.items():
        if key == "npus_count":
            network_config_for_parallelism[key] = [int(x) for x in dims[parallelism_position]]
        else:
            network_config_for_parallelism[key] = [value[i] for i in parallelism_index]

    network_config_for_parallelism["dimensions-count"] = len(dims[parallelism_position])
    return network_config_for_parallelism

import json

def replace_collective_implementation(file_path, new_value):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    data["all-reduce-implementation"] = new_value
    # data["all-to-all-implementation"] = new_value
    data["all-gather-implementation"] = new_value
    data["reduce-scatter-implementation"] = new_value
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage


def get_astrasim_collective_time(collective_size, collective_type, system:System=None,
                                network_config=None) -> dict:
    """
    collective_type: str: Should be one of the following: "ALLREDUCE"/"ALLTOALL"/"ALLGATHER"/"REDUCESCATTER"
    collective_size: int: Size of the collective operation in Bytes
    
    Returns: dict: A dictionary with the key as the system id and the value is latency in ns
    
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if network_config is not None:
        nodes = int(np.prod(network_config["npus_count"]))
    else:
        nodes = system.num_nodes
        
    assert collective_type in ["ALLREDUCE", "ALLTOALL", "ALLGATHER", "REDUCESCATTER"], \
        "Invalid collective_type. Must be one of: ALLREDUCE, ALLTOALL, ALLGATHER, REDUCESCATTER"
    # Step 1: Create the text file
    if collective_type == "ALLGATHER":
        ## For Allgather, astra-sim expects the collective size to be the size of the message sent by each node
        collective_size = int(collective_size/nodes)
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
    with open(txt_file_path, "w+") as txt_file:
        txt_file.write("MICRO\n1\nDUMMYNAME -1 5 NONE 0 5 NONE 0 5 {} {} 5\n".format(collective_type, collective_size))
    
    # Step 2: Generate ET trace
    os.makedirs(os.path.dirname(et_output_path), exist_ok=True)


    # Wait for the subprocess to complete
    with contextlib.redirect_stdout(io.StringIO()):
        result = subprocess.run([
            "python", "-m", "chakra.src.converter.converter", "Text",
            "--input", txt_file_path,
            "--output", et_output_path + "collective_traces",
            "--num-npus", str(nodes),
            "--num-passes", "1",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            print(result.stdout)
    
    # Step 3: Clean up the generated traces
    os.makedirs(et_cleaned_output_path, exist_ok=True)
    files = os.listdir(et_output_path)
    for file in files:
        if not file.endswith(".et"):
            continue
        input_file = os.path.join(et_output_path, file)
        output_file = os.path.join(et_cleaned_output_path, file)
        convert_chakra_file(input_file, output_file)
    
    # Step 4: Create network.yml
    if network_config is None:
        assert system.topology in ["Ring", "FullyConnected", "Switch"], \
            "Invalid collective_type. Must be one of: Ring, FullyConnected, Switch"
        network_config = {
            "topology": [system.topology],    #(“Ring”, “FullyConnected”, or “Switch”)
            "npus_count": [system.num_nodes],
            "bandwidth": [Unit().raw_to_unit(system.interchip_link_bw, type="BW")],
            "latency": [system.interchip_link_latency*1e9],## AStrasim expects latency in ns 
        }
    else:
        assert type(network_config) == dict, "network_config must be a dictionary"

    network_yml_path = SCRIPT_DIR+"/network.yml"
    with open(network_yml_path, "w+") as network_yml_file:
        yaml.dump(network_config, network_yml_file)
    
    # # Step 5: Replace the system implementation
    topology_to_algorithm = {
        "Ring": "ring",
        "FullyConnected": "direct",
        "Switch": "halvingDoubling"
    }
    collective_impl = [topology_to_algorithm[i] for i in network_config['topology']]
    replace_collective_implementation('/home/abhimanyu/synergy3/work/GenZ-LLM-Analyzer/GenZ/Astra_sim/system.json', collective_impl) 
    # Step 6: Run astra-sim
    result = subprocess.run(f"bash {run_file}>{ASTRA_SIM_OUTPUT_PATH}", shell=True, check=True, stderr=subprocess.PIPE)
    if result.stdout:
        print(result.stdout)

    def get_cycles_count(file_path):
        cycles_count = {}
        with open(file_path, 'r') as file:
            for line in file:
                if 'finished' in line:
                    parts = line.split(',')
                    sys_id = parts[0].split('[')[1].split(']')[0]
                    cycles = int(parts[1].split()[0])
                    cycles_count[sys_id] = cycles
        return cycles_count
    
    cycles_count = get_cycles_count(ASTRA_SIM_OUTPUT_PATH)
    return cycles_count