import pytest
from GenZ import prefill_moddeling, get_model_df, get_configs, System, create_inference_moe_prefill_layer, get_AR_time
import os
import pandas as pd
import numpy as np

def test_dense_LLM_prefill():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')
    Model = 'gpt-2'
    prefill_output = prefill_moddeling(model = Model, batch_size = 1, input_tokens = 4096,
                                system_name = TPU, bits='bf16', tensor_parallel = 1, pipeline_parallel = 1, debug=False)

    ref_latency = 5.476083302399999
    ref_throughput = 182.6122695324468
    ref_runtime_breakdown = [2.8991029248, 2.5769803776, 0.0]
    assert np.allclose([prefill_output['Latency'], prefill_output['Throughput'], prefill_output['Runtime_breakdown'][0], prefill_output['Runtime_breakdown'][1], prefill_output['Runtime_breakdown'][2]],
                        [ref_latency, ref_throughput, ref_runtime_breakdown[0], ref_runtime_breakdown[1], ref_runtime_breakdown[2]])

def test_dense_LLM_prefill_with_tensor_parallel():
    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16',
                interchip_mem_bw=50, interchip_link_latency=1)
    Model = 'gpt-2'
    prefill_output = prefill_moddeling(model = Model, batch_size = 1, input_tokens = 4096,
                                system_name = TPU, bits='bf16', tensor_parallel = 4, pipeline_parallel = 1, debug=False)

    ref_latency = 5.886358809914
    ref_throughput = 169.884309178665
    ref_runtime_breakdown = [0.759363715514, 0.6442450944, 4.48275]
    assert np.allclose([prefill_output['Latency'], prefill_output['Throughput'], prefill_output['Runtime_breakdown'][0], prefill_output['Runtime_breakdown'][1], prefill_output['Runtime_breakdown'][2]],
                        [ref_latency, ref_throughput, ref_runtime_breakdown[0], ref_runtime_breakdown[1], ref_runtime_breakdown[2]])