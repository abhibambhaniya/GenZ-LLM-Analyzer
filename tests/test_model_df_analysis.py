import pytest
from GenZ import get_model_df, get_summary_table, System, create_inference_moe_prefill_layer, create_inference_moe_decode_layer

import os
import pandas as pd

def test_dense_LLM_prefill():
    # Delete the current CSV file if it exists
    if os.path.exists('/tmp/current_llama2_7b_prefill_on_TPU.csv'):
        os.remove('/tmp/current_llama2_7b_prefill_on_TPU.csv')

    # Load the golden result
    golden_df = pd.read_csv('./golden/llama2_7b_prefill_on_TPU.csv')

    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')

    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_prefill_layer(1024, "llama2_7b"), system=TPU)
    current_df.to_csv('/tmp/current_llama2_7b_prefill_on_TPU.csv', index=False)

    # Reload the saved current result
    reloaded_current_df = pd.read_csv('/tmp/current_llama2_7b_prefill_on_TPU.csv')

    # Ensure the reloaded dataframe matches the original current dataframe
    pd.testing.assert_frame_equal(golden_df, reloaded_current_df)

def test_dense_LLM_decode():
    # Delete the current CSV file if it exists
    if os.path.exists('/tmp/current_llama2_7b_decode_on_TPU.csv'):
        os.remove('/tmp/current_llama2_7b_decode_on_TPU.csv')

    # Load the golden result
    golden_df = pd.read_csv('./golden/llama2_7b_decode_on_TPU.csv')

    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')

    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_decode_layer(1024, "llama2_7b"), system=TPU)
    current_df.to_csv('/tmp/current_llama2_7b_decode_on_TPU.csv', index=False)

    # Reload the saved current result
    reloaded_current_df = pd.read_csv('/tmp/current_llama2_7b_decode_on_TPU.csv')

    # Ensure the reloaded dataframe matches the original current dataframe
    pd.testing.assert_frame_equal(golden_df, reloaded_current_df)

def test_moe_LLM_prefill():
    # Delete the current CSV file if it exists
    if os.path.exists('/tmp/current_mixtral_8x7b_prefill_on_GH200.csv'):
        os.remove('/tmp/current_mixtral_8x7b_prefill_on_GH200.csv')

    # Load the golden result
    golden_df = pd.read_csv('./golden/mixtral_8x7b_prefill_on_GH200.csv')

    # Generate the current result
    GH200 = System(flops=2000, offchip_mem_bw=4900, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16',
                off_chip_mem_size=144)

    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_prefill_layer(1024, "mixtral_8x7b"), system=GH200)
    current_df.to_csv('/tmp/current_mixtral_8x7b_prefill_on_GH200.csv', index=False)

    # Reload the saved current result
    reloaded_current_df = pd.read_csv('/tmp/current_mixtral_8x7b_prefill_on_GH200.csv')

    # Ensure the reloaded dataframe matches the original current dataframe
    pd.testing.assert_frame_equal(golden_df, reloaded_current_df)

def test_moe_LLM_decode():
    # Delete the current CSV file if it exists
    if os.path.exists('/tmp/current_mixtral_8x7b_decode_on_GH200.csv'):
        os.remove('/tmp/current_mixtral_8x7b_decode_on_GH200.csv')

    # Load the golden result
    golden_df = pd.read_csv('./golden/mixtral_8x7b_decode_on_GH200.csv')

    # Generate the current result
    GH200 = System(flops=2000, offchip_mem_bw=4900, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16',
                off_chip_mem_size=144)

    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_decode_layer(1024, "mixtral_8x7b"), system=GH200)
    current_df.to_csv('/tmp/current_mixtral_8x7b_decode_on_GH200.csv', index=False)

    # Reload the saved current result
    reloaded_current_df = pd.read_csv('/tmp/current_mixtral_8x7b_decode_on_GH200.csv')

    # Ensure the reloaded dataframe matches the original current dataframe
    pd.testing.assert_frame_equal(golden_df, reloaded_current_df)