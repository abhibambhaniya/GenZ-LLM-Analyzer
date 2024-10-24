import pytest
from GenZ import prefill_moddeling, get_model_df, get_configs, System, create_inference_moe_prefill_layer, get_AR_time
import os
import pandas as pd

def test_dense_LLM_prefill():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')
    Model = 'gpt-2'
    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_prefill_layer(4096, Model), system=TPU)

    prefill_output = prefill_moddeling(model = Model, batch_size = 1, input_tokens = 4096,
                                system_name = TPU, bits='bf16', tensor_parallel = 1, pipeline_parallel = 1, debug=False)

    prefill_latency = prefill_output['Latency']

    assert prefill_latency == sum(current_df['Latency (msec)']) * get_configs(Model).num_decoder_layers

def test_dense_LLM_prefill_with_tensor_parallel():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16',
                interchip_mem_bw=50, interchip_link_latency=1)
    Model = 'gpt-2'
    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_prefill_layer(4096, Model, tensor_parallel=4), system=TPU)

    ## For GPT-2, the AR message size is 6 MB (4k tokens * 2 bytes)
    AR_time = get_AR_time(data = 6*2**20, num_AR_nodes = 4, system = TPU)

    prefill_output = prefill_moddeling(model = Model, batch_size = 1, input_tokens = 4096,
                                system_name = TPU, bits='bf16', tensor_parallel = 4, pipeline_parallel = 1, debug=False)

    prefill_latency = prefill_output['Latency']

    assert prefill_latency == (sum(current_df['Latency (msec)']) + 2 * AR_time )* get_configs(Model).num_decoder_layers