import pytest
from GenZ.Models.get_language_model import get_configs, create_inference_moe_prefill_layer, create_inference_moe_decode_layer
import os
import pandas as pd

MODEL_PATH = "/tmp/genz/data/model"

def test_get_configs():
    # Test known model names
    assert get_configs('gpt-2').model == 'openai/gpt2'
    assert get_configs('facebook/opt-125m').model == 'facebook/opt-125M'

def test_create_inference_dense_prefix_model():
    file_name = create_inference_moe_prefill_layer(input_sequence_length=10, name='gpt-2')
    assert file_name.endswith('.csv')
    assert 'gpt-2_prefix' in file_name
    df = pd.read_csv(os.path.join(MODEL_PATH, file_name), header=None)
    gpt2_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['2304','10','768','1','1','0','3'],
                ['12','10','10','64','12','3','4'],
                ['12','10','10','64','12','1','5'],
                ['768','10','768','1','1','0','3'],
                ['3072','10','768','1','1','0','3'],
                ['768','10','3072','1','1','0','3'],
])
    pd.testing.assert_frame_equal(gpt2_ref,df)

def test_create_inference_dense_gemma_prefix_model():
    file_name = create_inference_moe_prefill_layer(input_sequence_length=10, name='gemma2_9b')
    assert file_name.endswith('.csv')
    assert 'gemma2_9b_prefix' in file_name
    df = pd.read_csv(os.path.join(MODEL_PATH, file_name), header=None)
    gemma2_9b_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['8192','10','3584','1','1','0','3'],
                ['16','10','10','256','8','3','4'],
                ['16','10','10','256','8','1','5'],
                ['3584','10','4096','1','1','0','3'],
                ['28672','10','3584','1','1','0','3'],
                ['3584','10','14336','1','1','0','3'],
])
    pd.testing.assert_frame_equal(gemma2_9b_ref,df)

def test_create_inference_moe_prefill_layer():
    file_name = create_inference_moe_prefill_layer(input_sequence_length=10, name='mistralai/mixtral-8x7b')
    assert file_name.endswith('.csv')
    assert 'mixtral-8x7b_prefix' in file_name
    df = pd.read_csv(os.path.join(MODEL_PATH, file_name), header=None)
    mixtral_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['6144','10','4096','1','1','0','3'],
                ['32','10','10','128','8','3','4'],
                ['32','10','10','128','8','1','5'],
                ['4096','10','4096','1','1','0','3'],
                ['229376','2','4096','1','1','0','3'],
                ['4096','2','114688','1','1','0','3'],
    ])
    pd.testing.assert_frame_equal(mixtral_ref,df)

def test_create_inference_dense_decode_model():
    file_name = create_inference_moe_decode_layer(input_sequence_length=10, output_gen_tokens=32, name='gpt-2')
    assert file_name.endswith('.csv')
    assert 'gpt-2_decode' in file_name

    df = pd.read_csv(os.path.join(MODEL_PATH, file_name), header=None)
    gpt2_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['2304','1','768','1','1','5','3'],
                ['12','1','10','64','12','5','9'],
                ['12','1','32','64','12','5','4'],
                ['12','1','10','64','12','5','10'],
                ['12','1','32','64','12','5','5'],
                ['768','1','768','1','1','5','3'],
                ['3072','1','768','1','1','5','3'],
                ['768','1','3072','1','1','5','3'],])

    pd.testing.assert_frame_equal(gpt2_ref, df)

def test_create_inference_moe_decode_layer():
    file_name = create_inference_moe_decode_layer(input_sequence_length=10, output_gen_tokens=32, name='mistralai/mixtral-8x7b')
    assert file_name.endswith('.csv')
    assert 'mixtral-8x7b_decode' in file_name

    df = pd.read_csv(os.path.join(MODEL_PATH, file_name), header=None)
    mixtral_ref = pd.DataFrame([
        ['M','N','D','H','Z','Z','T'],
        ['6144','1','4096','1','1','5','3'],
        ['32','1','10','128','8','5','9'],
        ['32','1','32','128','8','5','4'],
        ['32','1','10','128','8','5','10'],
        ['32','1','32','128','8','5','5'],
        ['4096','1','4096','1','1','5','3'],
        ['57344','1','4096','1','1','5','3'],
        ['172032','0','4096','1','1','0','3'],
        ['4096','1','28672','1','1','5','3'],
        ['4096','0','86016','1','1','0','3'],
        ])

    pd.testing.assert_frame_equal(mixtral_ref, df)