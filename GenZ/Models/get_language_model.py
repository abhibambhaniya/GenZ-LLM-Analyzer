import pandas as pd
import os
from math import ceil
import numpy as np
from datetime import datetime
from GenZ.parallelism import ParallelismConfig

from GenZ.Models import ModelConfig, MODEL_DICT

from GenZ.Models.utils import OpType, ResidencyInfo, parse_einsum_expression
from GenZ.Models.attention import mha_flash_attention_prefill, mha_flash_attention_decode
from GenZ.Models.ffn import ffn_prefill, ffn_decode
from GenZ.Models.mamba import mamba_prefill, mamba_decode

def get_configs(name) -> ModelConfig:
    name = name.lower()

    if model := MODEL_DICT.get(name):
        model_config = model
    else:
        print("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

    return model_config

def save_layers(layers:str, data_path:str, name:str):
    model_path = os.path.join(data_path,"model")
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    file_name = name.replace("/", "_") + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") +'.csv'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df.to_csv(os.path.join(model_path, file_name),  header=True, index=None)
    return file_name


DATA_PATH = "/tmp/genz/data/"

def create_inference_moe_prefill_layer(input_sequence_length, name='BERT', data_path=DATA_PATH,
                         **args):
    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)

    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")

def create_inference_moe_decode_layer(input_sequence_length, name='BERT', data_path=DATA_PATH,
                         output_gen_tokens=32, **args):

    model_config = get_configs(name)
    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )
    layers = mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens) + ffn_decode(model_config, parallelism_config)

    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")

def create_full_prefill_model(input_sequence_length, name='BERT', data_path=DATA_PATH, **args):

    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)

    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")


def create_inference_mamba_prefix_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mamba_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)

    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")


def create_inference_mamba_decode_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)
    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )
    layers = mamba_decode(model_config, parallelism_config, input_sequence_length) + ffn_decode(model_config, parallelism_config)

    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")



def einsum_test(equation=None, einsum_vars=None):

    if equation is None:
        A = (2, 3, 4)
        B = (2, 4, 5)
        C = (5, 6)
        equation = 'ijk,ikl,lm->ijm'
        einsum_vars = parse_einsum_expression(equation, A, B, C)

    layers = [[equation, einsum_vars, 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]

    return save_layers(layers=layers, data_path=DATA_PATH, name="einsum_")
