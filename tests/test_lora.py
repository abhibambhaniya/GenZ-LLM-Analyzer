import pytest
import copy # To safely modify config copies for tests
from GenZ.LLM_inference.llm_decode import decode_moddeling
from GenZ.Models.get_language_model import get_configs
from GenZ.Models.default_models import ModelConfig, MODEL_DICT # To access MODEL_DICT for cleanup if necessary

# Define the test model name used in default_models.py
TEST_MODEL_NAME = "lora_test_model"

@pytest.fixture
def model_config():
    """Fixture to get the test model config and ensure lora_rank is reset."""
    config = get_configs(TEST_MODEL_NAME)
    original_lora_rank = config.lora_rank
    original_lora_alpha = config.lora_alpha
    yield config
    # Teardown: restore original LORA parameters
    config.lora_rank = original_lora_rank
    config.lora_alpha = original_lora_alpha

def test_lora_memory_overhead(model_config: ModelConfig):
    """Tests the memory overhead calculation with LORA."""
    # Ensure lora_rank is 0 for the baseline
    model_config.lora_rank = 0
    model_config.lora_alpha = 1 # Default alpha

    # Run decode_moddeling with LORA disabled
    # Using small token numbers for speed, system_eff=1 for predictability
    # Use model_profilling=True to get the summary_table with weights
    _, summary_table_no_lora = decode_moddeling(
        model=TEST_MODEL_NAME,
        batch_size=1,
        input_tokens=32,
        output_tokens=1,
        bits='bf16',
        system_eff=1,
        model_profilling=True
    )
    # Extract model weights
    weight_col_name_no_lora = [col for col in summary_table_no_lora.columns if "Total Weights" in col][0]
    weights_no_lora = summary_table_no_lora[weight_col_name_no_lora].values[0]

    # Enable LORA
    test_lora_rank = 8
    model_config.lora_rank = test_lora_rank
    
    _, summary_table_with_lora = decode_moddeling(
        model=TEST_MODEL_NAME, # Relies on get_configs picking up the modified model_config
        batch_size=1,
        input_tokens=32,
        output_tokens=1,
        bits='bf16',
        system_eff=1,
        model_profilling=True
    )
    weight_col_name_with_lora = [col for col in summary_table_with_lora.columns if "Total Weights" in col][0]
    weights_with_lora = summary_table_with_lora[weight_col_name_with_lora].values[0]

    # Calculate expected LORA weights
    bytes_per_param = 2  # for bf16
    
    # Parameters from lora_test_model_config in default_models.py
    hidden_size = model_config.hidden_size
    head_dim = model_config.head_dim
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    num_decoder_layers = model_config.num_decoder_layers

    q_lora_a = hidden_size * test_lora_rank * bytes_per_param
    q_lora_b = test_lora_rank * (head_dim * num_attention_heads) * bytes_per_param
    v_lora_a = hidden_size * test_lora_rank * bytes_per_param
    v_lora_b = test_lora_rank * (head_dim * num_key_value_heads) * bytes_per_param
    
    total_lora_per_layer = q_lora_a + q_lora_b + v_lora_a + v_lora_b
    total_lora_weights_bytes = num_decoder_layers * total_lora_per_layer
    expected_lora_weights_mb = total_lora_weights_bytes / (1024 * 1024)

    # Assert that model_weights with LORA is approximately weights_no_lora + expected_lora_weights_mb
    # Allow for small floating point inaccuracies
    assert weights_with_lora == pytest.approx(weights_no_lora + expected_lora_weights_mb, rel=1e-3)


def test_lora_latency_and_ops(model_config: ModelConfig):
    """Tests latency increase and presence of LORA ops."""
    model_config.lora_rank = 0
    model_config.lora_alpha = 1

    # --- Check ops and latency with LORA disabled ---
    model_config.lora_rank = 0
    model_config.lora_alpha = 1

    # Get model_df for op checking (model_profilling=True returns (df, summary_table))
    df_no_lora, _ = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', system_eff=1, model_profilling=True
    )
    assert df_no_lora is not None, "model_df should not be None when model_profilling=True for no_lora case"
    
    # Get latency (model_profilling=False returns ModdelingOutput object)
    output_no_lora_runtime = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', system_eff=1 # model_profilling is False by default
    )
    latency_no_lora = output_no_lora_runtime.Latency
    
    # Check for absence of LORA ops
    lora_op_names = {"LORA_A_Q", "LORA_B_Q", "LORA_A_V", "LORA_B_V"}
    # Operator name column is "Layer Name"
    present_ops_no_lora = set(df_no_lora["Layer Name"].unique())
    assert len(lora_op_names.intersection(present_ops_no_lora)) == 0, \
        f"LORA ops found when lora_rank=0: {lora_op_names.intersection(present_ops_no_lora)}"

    # --- Check ops and latency with LORA enabled ---
    model_config.lora_rank = 8
    
    # Get model_df for op checking
    df_with_lora, _ = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', system_eff=1, model_profilling=True
    )
    assert df_with_lora is not None, "model_df should not be None when model_profilling=True for with_lora case"

    # Get latency
    output_with_lora_runtime = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', system_eff=1 # model_profilling is False by default
    )
    latency_with_lora = output_with_lora_runtime.Latency

    assert latency_with_lora >= latency_no_lora, "Latency with LORA should be greater than or equal to without LORA"

    # Check for presence of LORA ops
    # Operator name column is "Layer Name"
    present_ops_with_lora = set(df_with_lora["Layer Name"].unique())
    assert lora_op_names.issubset(present_ops_with_lora), \
        f"Expected LORA ops not found when lora_rank=8. Missing: {lora_op_names - present_ops_with_lora}"


def test_lora_with_tensor_parallel(model_config: ModelConfig):
    """Tests LORA functionality with tensor parallelism."""
    tp_val = 2
    model_config.lora_rank = 0
    model_config.lora_alpha = 1

    # --- Check ops and latency with LORA disabled and TP ---
    model_config.lora_rank = 0
    model_config.lora_alpha = 1

    # Get model_df for op checking
    df_no_lora_tp, _ = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', tensor_parallel=tp_val, system_eff=1, model_profilling=True
    )
    assert df_no_lora_tp is not None, "model_df should not be None for no_lora_tp case"

    # Get latency
    output_no_lora_tp_runtime = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', tensor_parallel=tp_val, system_eff=1 # model_profilling is False
    )
    latency_no_lora_tp = output_no_lora_tp_runtime.Latency

    lora_op_names = {"LORA_A_Q", "LORA_B_Q", "LORA_A_V", "LORA_B_V"}
    # Operator name column is "Layer Name"
    present_ops_no_lora_tp = set(df_no_lora_tp["Layer Name"].unique())
    assert len(lora_op_names.intersection(present_ops_no_lora_tp)) == 0, \
        f"LORA ops found when lora_rank=0 with TP: {lora_op_names.intersection(present_ops_no_lora_tp)}"

    # --- Check ops and latency with LORA enabled and TP ---
    model_config.lora_rank = 4 # Use a different rank for variation
    
    # Get model_df for op checking
    df_with_lora_tp, _ = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', tensor_parallel=tp_val, system_eff=1, model_profilling=True
    )
    assert df_with_lora_tp is not None, "model_df should not be None for with_lora_tp case"

    # Get latency
    output_with_lora_tp_runtime = decode_moddeling(
        model=TEST_MODEL_NAME, batch_size=1, input_tokens=128, output_tokens=1, # Further Increased input_tokens
        bits='bf16', tensor_parallel=tp_val, system_eff=1 # model_profilling is False
    )
    latency_with_lora_tp = output_with_lora_tp_runtime.Latency
    
    assert latency_with_lora_tp >= latency_no_lora_tp, "Latency with LORA and TP should be greater than or equal to without LORA with TP"

    # Operator name column is "Layer Name"
    present_ops_with_lora_tp = set(df_with_lora_tp["Layer Name"].unique())
    assert lora_op_names.issubset(present_ops_with_lora_tp), \
        f"Expected LORA ops not found when lora_rank=4 with TP. Missing: {lora_op_names - present_ops_with_lora_tp}"

# It might be good to add a test that checks if lora_alpha influences calculations,
# but the current design primarily focuses on ops and their sizes, not scaled values directly in summary tables.
# For now, we assume lora_alpha is used internally by the GEMM ops if needed, but its effect isn't directly testable via summary table weights/latency.

# To run these tests:
# Ensure GenZ and its dependencies are in PYTHONPATH
# Navigate to the root of the GenZ repo
# Run: pytest tests/test_lora.py
# Make sure a system like 'A100_40GB_GPU' is available or use a default/dummy system for tests if decode_moddeling allows.
# The decode_moddeling by default uses 'A100_40GB_GPU', bits='bf16', system_eff=1.
# If tests are slow, ensure input_tokens/output_tokens are small.
# The `lora_test_model` is small, so it should be reasonably fast.
# The `system_eff=1` makes calculations more predictable by removing efficiency scaling.
# If `decode_moddeling` requires a specific system to be configured, that might need setup.
# For now, assume default system works.
# The tests rely on `get_configs` returning a mutable config object that is shared/cached,
# so changes to `model_config.lora_rank` are seen by `decode_moddeling`.
# The fixture ensures `lora_rank` is reset after each test.
# Note on weight column name: The test dynamically finds the weight column.
# This makes it resilient to exact naming like "(MB)" vs "(GB)" as long as "Total Weights" is in the name.
# If unit.py changes unit representation, this part might need adjustment.
# For simplicity, tests use default `system_name='A100_40GB_GPU'` and `bits='bf16'`.
# If a simpler/faster system is available for tests, it could be specified.
# The 'lora_test_model' has D=128, L=2, H=4, Hkv=2, Dq=32.
# With lora_rank=8, bytes_per_param=2:
# q_lora_a = 128 * 8 * 2 = 2048
# q_lora_b = 8 * (32 * 4) * 2 = 8 * 128 * 2 = 2048
# v_lora_a = 128 * 8 * 2 = 2048
# v_lora_b = 8 * (32 * 2) * 2 = 8 * 64 * 2 = 1024
# total_lora_per_layer = 2048 + 2048 + 2048 + 1024 = 7168 bytes
# total_lora_weights_bytes = 2 * 7168 = 14336 bytes
# expected_lora_weights_mb = 14336 / (1024*1024) = 0.013671875 MB
# This value will be used in test_lora_memory_overhead.
