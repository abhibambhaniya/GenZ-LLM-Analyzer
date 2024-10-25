from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
from GenZ.analyse_model import *
import warnings
from GenZ.LLM_inference import decode_moddeling, prefill_moddeling
from paretoset import paretoset
import itertools
from GenZ.Models import get_configs, create_inference_moe_prefill_layer, create_inference_moe_decode_layer

unit = Unit()

def factors(n):
    return [x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup]

def get_various_parallization(model='llama2_7b', total_nodes=8):
    model_config = get_configs(model)

    if total_nodes == 1:
        return {(1,1)}
    elif total_nodes < 1:
        raise ValueError(f'Num of Nodes:{total_nodes} should be >= 1')

    H = model_config.num_attention_heads
    num_layers = model_config.num_decoder_layers

    TP_parallelism = np.sort(factors(H))[::-1]
    PP_parallelism = np.sort(factors(num_layers))[::-1]

    parallelism_combinations = set()

    for TP, PP in itertools.product(TP_parallelism, PP_parallelism):
        if TP * PP < total_nodes and TP*PP >= total_nodes//2:
            parallelism_combinations.add((TP, PP))
    return parallelism_combinations

def get_best_parallization_strategy(
        stage='decode', model='llama2_7b', total_nodes=8, batch_size = 1, beam_size = 1,
        input_tokens = 2000, output_tokens = 256,
        system_name = {'Flops': 200, 'Memory_size': 32, 'Memory_BW': 1000, 'ICN': 300 , 'real_values':True},
        bits='bf16', debug=False
        ):

    parallelism_combinations = get_various_parallization(model=model, total_nodes=total_nodes)
    if debug:
        print(f'For model:{model}, number cores:{total_nodes}, system:{system_name}, \n The parallelism combinatations are {parallelism_combinations} ')

    data = []
    for TP,PP in parallelism_combinations:
        if PP <= batch_size:
            micro_batch_size = (batch_size//PP)
            if stage == 'prefill':
                prefill_outputs = prefill_moddeling(model = model, batch_size = micro_batch_size,
                                        input_tokens = input_tokens,
                                        system_name = system_name,
                                        bits=bits,
                                        tensor_parallel = TP, pipeline_parallel = PP, debug=debug)
                data.append([micro_batch_size, TP, PP , prefill_outputs['Latency'], prefill_outputs['Throughput']] + prefill_outputs['Runtime_breakdown'])
            elif stage == 'decode':
                decode_outputs = decode_moddeling(model = model, batch_size = micro_batch_size, Bb = beam_size ,
                                    input_tokens = input_tokens, output_tokens = output_tokens, 
                                    system_name = system_name,
                                    bits=bits,
                                    tensor_parallel = TP, pipeline_parallel =PP, debug=debug)
                data.append([micro_batch_size, TP, PP,  decode_outputs['Latency'], decode_outputs['Throughput']] + decode_outputs['Runtime_breakdown'])
            else:
                raise ValueError('Stage should be prefill or decode')

    data_df = pd.DataFrame(data, columns = ['micro batch', 'TP', 'PP', 'Latency(ms)', 'Tokens/s', 'GEMM time', 'SA time', 'Comm. time'])
    if debug:
        display(data_df)
    return data_df.sort_values(by='Tokens/s', ascending=False).head(1)

def get_pareto_optimal_performance(
        stage='decode', model='llama2_7b', total_nodes=8, batch_list = 1, beam_size = 1,
        input_tokens = 2000, output_tokens = 256,
        system_name = {'Flops': 200, 'Memory_size': 32, 'Memory_BW': 1000, 'ICN': 300 , 'real_values':True},
        bits='bf16', debug=False
        ):

    parallelism_combinations = get_various_parallization(model=model, total_nodes=total_nodes)

    if debug:
        print(f'For model:{model}, number cores:{total_nodes}, system:{system_name}, \n The parallelism combinatations are {parallelism_combinations} ')
    data = []
    if isinstance(batch_list, int):
        batch_list = [batch_list]
    for batch_size in batch_list:
        for TP,PP in parallelism_combinations:
            if PP <= batch_size:
                micro_batch_size = (batch_size//PP)
                if stage == 'prefill':
                    prefill_outputs = prefill_moddeling(model = model, batch_size = micro_batch_size,
                                            input_tokens = input_tokens,
                                            system_name = system_name,
                                            bits=bits,
                                            tensor_parallel = TP, pipeline_parallel = PP, debug=False)
                    data.append([batch_size, micro_batch_size, TP, PP , prefill_outputs['Latency'], prefill_outputs['Throughput']] + prefill_outputs['Runtime_breakdown'])
                elif stage == 'decode':
                    decode_outputs = decode_moddeling(model = model, batch_size = micro_batch_size, Bb = beam_size ,
                                        input_tokens = input_tokens, output_tokens = output_tokens, 
                                        system_name = system_name,
                                        bits=bits,
                                        tensor_parallel = TP, pipeline_parallel =PP, debug=False)
                    data.append([batch_size, micro_batch_size, TP, PP,  decode_outputs['Latency'], decode_outputs['Throughput']] + decode_outputs['Runtime_breakdown'])
                else:
                    raise ValueError('Stage should be prefill or decode')

    data_df = pd.DataFrame(data, columns = ['batch', 'micro batch', 'TP', 'PP', 'Latency(ms)', 'Tokens/s', 'GEMM time', 'SA time', 'Comm. time'])
    datapoints = data_df[['Latency(ms)','Tokens/s']]

    ##  We want a pareto optimal frontier with minimum latency and maximum Throughput.
    mask = paretoset(datapoints, sense=["min", "max"])
    return data_df[mask]