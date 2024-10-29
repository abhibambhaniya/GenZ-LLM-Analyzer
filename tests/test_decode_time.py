import pytest
from GenZ import decode_moddeling, get_model_df, get_configs, System, create_inference_moe_decode_layer, get_AR_time
import os
import pandas as pd

def test_dense_LLM_decode():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')
    Model = 'gpt-2'
    Bb = 4
    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_decode_layer(4096, Model, output_gen_tokens=100), system=TPU
                            , batch_size=Bb, beam_merge= (Bb > 1), beam_size= Bb)

    decode_output = decode_moddeling(model = Model, batch_size = 1, input_tokens = 4096, output_tokens=100, Bb=Bb,
                                system_name = TPU, bits='bf16', tensor_parallel = 1, pipeline_parallel = 1, debug=False)

    decode_latency = decode_output['Latency']

    assert decode_latency == sum(current_df['Latency (msec)']) * get_configs(Model).num_decoder_layers
    
def test_dense_LLM_decode_with_tensor_parallel():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16',
                interchip_link_bw=50, interchip_link_latency=1)
    Model = 'gpt-2'
    Bb = 4
    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_decode_layer(4096, Model, tensor_parallel=4, output_gen_tokens=1024), system=TPU
                            , batch_size=Bb, beam_merge= (Bb > 1), beam_size= Bb)
    
    ## For GPT-2, the AR message size is 6 KB
    AR_time = get_AR_time(data = 6*2**10, num_AR_nodes = 4, system = TPU)

    decode_output = decode_moddeling(model = Model, batch_size = 1, input_tokens = 4096, output_tokens=1024, Bb=Bb, 
                                system_name = TPU, bits='bf16', tensor_parallel = 4, pipeline_parallel = 1, debug=False)

    decode_latency = decode_output['Latency']

    assert decode_latency == (sum(current_df['Latency (msec)']) + 2 * AR_time )* get_configs(Model).num_decoder_layers