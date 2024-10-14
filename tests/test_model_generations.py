import pytest
from GenZ.get_language_model import get_configs, create_inference_moe_prefix_model, create_inference_moe_decode_model
import os
import pandas as pd

def test_get_configs():
    # Test known model names
    assert get_configs('gpt-2').model == 'openai/gpt2'
    assert get_configs('facebook/opt-125m').model == 'facebook/opt-125M'

def test_create_inference_moe_prefix_model():
    file_name = create_inference_moe_prefix_model(input_sequence_length=10, name='gpt-2')
    assert file_name.endswith('.csv')
    assert 'gpt-2_prefix' in file_name
    df = pd.read_csv(os.path.join('/tmp/data/model', file_name), header=None)
    gpt2_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['2304','10','768','1','1','1','3'],
                ['12','10','10','64','12','1','4'],
                ['12','10','10','64','12','1','5'],
                ['768','10','768','1','1','1','3'],
                ['3072','10','768','1','1','1','3'],
                ['768','10','3072','1','1','1','3']])
    are_equal = gpt2_ref.equals(df)
    assert are_equal

    file_name = create_inference_moe_prefix_model(input_sequence_length=10, name='mistralai/mixtral-8x7b')
    assert file_name.endswith('.csv')
    assert 'mixtral-8x7b_prefix' in file_name
    df = pd.read_csv(os.path.join('/tmp/data/model', file_name), header=None)
    mixtral_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['6144','10','4096','1','1','1','3'],
                ['32','10','10','128','8','1','7'],
                ['32','10','10','128','8','1','8'],
                ['4096','10','4096','1','1','1','3'],
                ['114688','2','4096','1','1','1','3'],
                ['114688','2','4096','1','1','1','3'],
                ['4096','2','114688','1','1','1','3']])
    are_equal = mixtral_ref.equals(df)
    assert are_equal

def test_create_inference_moe_decode_model():
    file_name = create_inference_moe_decode_model(input_sequence_length=10, output_gen_tokens=32, name='gpt-2')
    assert file_name.endswith('.csv')
    assert 'gpt-2_decode' in file_name

    df = pd.read_csv(os.path.join('/tmp/data/model', file_name), header=None)
    gpt2_ref = pd.DataFrame([
                ['M','N','D','H','Z','Z','T'],
                ['2304','1','768','1','1','1','3'],
                ['12','1','10','64','12','1','9'],
                ['12','1','22','64','12','1','4'],
                ['12','1','10','64','12','1','10'],
                ['12','1','22','64','12','1','5'],
                ['768','1','768','1','1','1','3'],
                ['3072','1','768','1','1','1','3'],
                ['768','1','3072','1','1','1','3']])

    are_equal = gpt2_ref.equals(df)
    assert are_equal

    file_name = create_inference_moe_decode_model(input_sequence_length=10, output_gen_tokens=32, name='mistralai/mixtral-8x7b')
    assert file_name.endswith('.csv')
    assert 'mixtral-8x7b_decode' in file_name
    df = pd.read_csv(os.path.join('/tmp/data/model', file_name), header=None)
    mixtral_ref = pd.DataFrame([
        ['M','N','D','H','Z','Z','T'],
        ['6144','1','4096','1','1','1','3'],
        ['32','1','10','128','8','1','7'],
        ['32','1','22','128','8','1','7'],
        ['32','1','10','128','8','1','8'],
        ['32','1','22','128','8','1','8'],
        ['4096','1','4096','1','1','1','3'],
        ['28672','1','4096','1','1','1','3'],
        ['86016','0','4096','1','1','1','3'],
        ['28672','1','4096','1','1','1','3'],
        ['86016','0','4096','1','1','1','3'],
        ['4096','1','28672','1','1','1','3'],
        ['4096','0','86016','1','1','1','3']])

    assert  mixtral_ref.equals(df)
