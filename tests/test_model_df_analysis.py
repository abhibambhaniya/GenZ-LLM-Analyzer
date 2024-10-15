import pytest
from GenZ import get_model_df, get_summary_table, System, create_inference_moe_prefix_model

import os
import pandas as pd

def test_LLM_prefill():
    # Delete the current CSV file if it exists
    if os.path.exists('/tmp/current_llama_7b_prefix_on_TPU.csv'):
        os.remove('/tmp/current_llama_7b_prefix_on_TPU.csv')

    # Load the golden result
    golden_df = pd.read_csv('./golden/llama_7b_prefix_on_TPU.csv')

    # Generate the current result
    TPU = System(flops=300, offchip_mem_bw=1200, compute_efficiency=0.8, memory_efficiency=0.8, bits='bf16')

    # Save the current result to a CSV file
    current_df = get_model_df(model=create_inference_moe_prefix_model(1024, "llama_7b"), system=TPU)
    current_df.to_csv('/tmp/current_llama_7b_prefix_on_TPU.csv', index=False)

    # Reload the saved current result
    reloaded_current_df = pd.read_csv('/tmp/current_llama_7b_prefix_on_TPU.csv')

    # Ensure the reloaded dataframe matches the original current dataframe
    pd.testing.assert_frame_equal(golden_df, reloaded_current_df)