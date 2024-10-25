import pandas as pd
import os
from math import ceil
import numpy as np
from datetime import datetime
from GenZ.parallelism import ParallelismConfig

from GenZ.Models.default_models import ModelConfig, MODEL_DICT

from GenZ.Models.utils import OpType, ResidencyInfo, CollectiveType, parse_einsum_expression
from GenZ.Models.attention import mha_flash_attention_prefill, mha_flash_attention_decode
from GenZ.Models.ffn import ffn_prefill, ffn_decode
from GenZ.Models.mamba import mamba_prefill, mamba_decode

def get_configs(name) -> ModelConfig:
    name = name.lower()

    if model := MODEL_DICT.get_model(name):
        model_config = model
    else:
        print("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

    return model_config

def save_layers(layers:list, data_path:str, name:str):
    model_path = os.path.join(data_path,"model")
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    file_name = name.replace("/", "_") + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") +'.csv'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df.to_csv(os.path.join(model_path, file_name),  header=True, index=None)
    return file_name

def repeat_layers(num_repeat:int):
    return [[num_repeat, 1, 1, 1, 1, 1, OpType.REPEAT]]

def end_repeat_layers(num_repeat:int):
    return [[num_repeat, 1, 1, 1, 1, 1, OpType.ENDREPEAT]]

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
    pipeline_stages = args.get('pipeline_parallel',1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1),
        )

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        layers += mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length)
        layers += ffn_prefill(model_config, parallelism_config, input_sequence_length)
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []

    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        ## For PP stages
        ## First PP-1 stages will have layers_per_stage layers and message pass at the end
        full_model += repeat_layers(pipeline_stages - 1)
        ## Single stage will have layers_per_stage layers
        full_model = add_layers(full_model, layers_per_stage)
        ## Single stage layers end and message pass at the end
        full_model += [[input_sequence_length // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        ## Last stage will have layers_last_stage layers and no message pass at the end
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    return save_layers(layers=full_model, data_path=data_path, name=name + "_prefix_")


def create_full_decode_model(input_sequence_length, name='BERT', data_path=DATA_PATH, output_gen_tokens=1, **args):

    model_config = get_configs(name)
    pipeline_stages = args.get('pipeline_parallel', 1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel', 1),
        expert_parallel=args.get('expert_parallel', 1),
        sequence_parallel=args.get('sequence_parallel', 1),
        data_parallel=args.get('data_parallel', 1),
    )

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        layers += mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens)
        layers += ffn_decode(model_config, parallelism_config)
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []

    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        full_model += repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_per_stage)
        full_model += [[input_sequence_length // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    return save_layers(layers=full_model, data_path=data_path, name=name + "_decode_")

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
