import pandas as pd
import os
from math import ceil
import numpy as np
from datetime import datetime
from GenZ.parallelism import ParallelismConfig

from GenZ.Models.default_models import ModelConfig, MODEL_DICT

from GenZ.Models.utils import OpType, ResidencyInfo, CollectiveType, parse_einsum_expression
from GenZ.Models.attention import mha_flash_attention_prefill, mha_flash_attention_decode, mha_flash_attention_chunked
from GenZ.Models.ffn import ffn_prefill, ffn_decode, deepseek_ffn_prefill
from GenZ.Models.mamba import mamba_prefill, mamba_decode
from GenZ.Models.embedding import input_embedding, output_embedding
from difflib import get_close_matches
from uuid import uuid4

def get_configs(name) -> ModelConfig:
    if isinstance(name, ModelConfig):
        return name
    elif isinstance(name, str):
        name = name.lower()

        if model := MODEL_DICT.get_model(name):
            model_config = model
            return model_config
        else:
            model_list = MODEL_DICT.list_models()
            close_matches = get_close_matches(name, model_list, cutoff=0.4)
            if close_matches:
                print("Did you mean one of these models?")
                for match in close_matches:
                    print(f" - {match}")
            raise ValueError("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

    else:
        raise ValueError("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

def get_ffn_implementation(model_config:ModelConfig):
    if model_config.ffn_implementation == "default":
        return ffn_prefill
    elif model_config.ffn_implementation == "deepseek":
        return deepseek_ffn_prefill
    else:
        raise ValueError("FFN implementation not supported")


def save_layers(layers:list, data_path:str, name:str):
    model_path = os.path.join(data_path,"model")
    df = pd.DataFrame(layers, columns=['Name', 'M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    file_name = name.replace("/", "_") + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + str(uuid4()) +'.csv'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df.to_csv(os.path.join(model_path, file_name),  header=True, index=None)
    return file_name

def repeat_layers(num_repeat:int):
    return [["Repeat", num_repeat, 1, 1, 1, 1, 1, OpType.REPEAT]]

def end_repeat_layers(num_repeat:int):
    return [["End Repeat", num_repeat, 1, 1, 1, 1, 1, OpType.ENDREPEAT]]

DATA_PATH = "/tmp/genz/data/"

def create_inference_moe_prefill_layer(input_sequence_length, name='GPT-2', data_path=DATA_PATH,
                         **args):
    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")

def create_inference_moe_decode_layer(input_sequence_length, name='GPT-2', data_path=DATA_PATH,
                         output_gen_tokens=32, **args):

    model_config = get_configs(name)
    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )
    layers = mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens) + ffn_decode(model_config, parallelism_config)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")

def create_full_prefill_model(
    name: str|ModelConfig ='GPT-2', 
    input_sequence_length: int=1024,
    data_path:str=DATA_PATH,
    **args) -> str:
    """
    The function `create_full_prefill_model` constructs a model with specified configurations and
    parallelism settings, saving the layers to a specified data path.
    
    name: The `name` parameter in the `create_full_prefill_model` function is used to specify the
    model configuration to be used. It can either be a string representing the name of the model
    (default is 'GPT-2') or an instance of `ModelConfig` class, defaults to GPT-2
    
    input_sequence_length: The `input_sequence_length` parameter specifies the length of the
    input sequence for the model. In this function, it is set to a default value of 1024. This parameter
    determines how many tokens or elements can be processed in a single input sequence, defaults to 1024
    
    data_path: The `data_path` parameter in the `create_full_prefill_model` function is a string
    that represents the path where the data will be saved or loaded from. It is a default parameter with
    a value of `DATA_PATH`, which is likely a constant or variable defined elsewhere in your codebase

    tensor_parallel: The `tensor_parallel` to define the degree of tensor parallelism, defaults to 1
    expert_parallel: The `expert_parallel` to define the degree of expert parallelism, defaults to 1
    sequence_parallel: The `sequence_parallel` to define the degree of sequence parallelism, defaults to 1
    data_parallel: The `data_parallel` to define the degree of data parallelism, defaults to 1
    pipeline_parallel: The `pipeline_parallel` to define the degree of pipeline parallelism, defaults to 1 
    
    return: The function `create_full_prefill_model` returns a string, which is the result of calling
    the `save_layers` function with the `full_model`, `data_path`, and a modified `name` as arguments.
    """
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
        if model_config.unique_layers == 1:
            if model_config.layer_type[0][0] == "MHA-global":
                layers += mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length)
            elif model_config.layer_type[0][0] == "Mamba":
                layers += mamba_prefill(model_config, parallelism_config, input_sequence_length)
        else:
            raise ValueError("More then 1 unique layers not supported. Work in progress")
        layers += ffn_prefill(model_config, parallelism_config, input_sequence_length)
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []
    full_model += input_embedding(model_config, parallelism_config, input_sequence_length)
    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        ## For PP stages
        ## First PP-1 stages will have layers_per_stage layers and message pass at the end
        full_model += repeat_layers(pipeline_stages - 1)
        ## Single stage will have layers_per_stage layers
        full_model = add_layers(full_model, layers_per_stage)
        ## Single stage layers end and message pass at the end
        full_model += [["Message Pass", input_sequence_length // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        ## Last stage will have layers_last_stage layers and no message pass at the end
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    full_model += output_embedding(model_config, parallelism_config, input_sequence_length)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name + "_prefix_")


def create_full_decode_model(
    name: str|ModelConfig ='GPT-2',
    input_sequence_length: int = 1024,
    output_gen_tokens: int = 0,
    data_path: str=DATA_PATH,
    **args) -> str:
    """
    The function `create_full_decode_model` constructs a decode model with specified configurations and
    parallelism settings, saving the layers to a specified data path.

    name: The `name` parameter in the `create_full_decode_model` function is used to specify the
    model configuration to be used. It can either be a string representing the name of the model
    (default is 'GPT-2') or an instance of `ModelConfig` class, defaults to GPT-2

    input_sequence_length: The `input_sequence_length` parameter specifies the length of the
    input sequence for the model. In this function, it is set to a default value of 1024. This parameter
    determines how many tokens or elements can be processed in a single input sequence, defaults to 1024

    output_gen_tokens: The `output_gen_tokens` parameter specifies the number of tokens to generated since the prefill.
                        This is to keep a track of multiple beams. Defaults to 1

    data_path: The `data_path` parameter in the `create_full_decode_model` function is a string
    that represents the path where the data will be saved or loaded from. It is a default parameter with
    a value of `DATA_PATH`, which is likely a constant or variable defined elsewhere in your codebase

    tensor_parallel: The `tensor_parallel` to define the degree of tensor parallelism, defaults to 1
    expert_parallel: The `expert_parallel` to define the degree of expert parallelism, defaults to 1
    sequence_parallel: The `sequence_parallel` to define the degree of sequence parallelism, defaults to 1
    data_parallel: The `data_parallel` to define the degree of data parallelism, defaults to 1
    pipeline_parallel: The `pipeline_parallel` to define the degree of pipeline parallelism, defaults to 1

    return: The function `create_full_decode_model` returns a string, which is the result of calling
    the `save_layers` function with the `full_model`, `data_path`, and a modified `name` as arguments.
    """
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
        if model_config.unique_layers == 1:
            if model_config.layer_type[0][0] == "MHA-global":
                layers += mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens)
            elif model_config.layer_type[0][0] == "Mamba":
                layers += mamba_decode(model_config, parallelism_config, input_sequence_length)
        else:
            raise ValueError("More then 1 unique layers not supported. Work in progress")
        layers += ffn_decode(model_config, parallelism_config)
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []

    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        full_model += repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_per_stage)
        full_model += [["Message Pass", 1, model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)
    full_model += output_embedding(model_config, parallelism_config, 1)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name + "_decode_")

def create_full_chunked_model(name:str ='GPT-2',
                            prefill_kv_sizes:list[(int,int)] =[], decode_kv_sizes: list[int]=[],
                            data_path:str = DATA_PATH, **args):
    ## Prefill KV sizes is a list of request by request, num tokens calculated and to be calculated.
    model_config = get_configs(name)
    pipeline_stages = args.get('pipeline_parallel',1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1),
        )

    ## Calculate the chunk size
    prefill_length = sum([i[1] for i in prefill_kv_sizes])
    chunk_size = len(decode_kv_sizes) + prefill_length

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        layers += mha_flash_attention_chunked(  model_config=model_config,
                                                parallelism_config=parallelism_config,
                                                chunk_size=chunk_size,
                                                prefill_kv_sizes=prefill_kv_sizes,
                                                decode_kv_sizes=decode_kv_sizes)
        layers += get_ffn_implementation(model_config)(model_config, parallelism_config, chunk_size)
        layers += end_repeat_layers(num_layers)
        return layers

    # assert prefill_length > 0, "Chunk size should be greater than the decode batches"
    full_model = []
    full_model += input_embedding(model_config, parallelism_config, prefill_length)
    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        ## For PP stages
        ## First PP-1 stages will have layers_per_stage layers and message pass at the end
        full_model += repeat_layers(pipeline_stages - 1)
        ## Single stage will have layers_per_stage layers
        full_model = add_layers(full_model, layers_per_stage)
        ## Single stage layers end and message pass at the end
        full_model += [["Message Pass", chunk_size // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        ## Last stage will have layers_last_stage layers and no message pass at the end
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    full_model += output_embedding(model_config, parallelism_config, chunk_size)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name+"_chunked_")


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
    if isinstance(name, ModelConfig):
        name = name.model
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
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")



def einsum_test(equation=None, einsum_vars=None):

    if equation is None:
        A = (2, 3, 4)
        B = (2, 4, 5)
        C = (5, 6)
        equation = 'ijk,ikl,lm->ijm'
        einsum_vars = parse_einsum_expression(equation, A, B, C)

    layers = [["test", equation, einsum_vars, 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]

    return save_layers(layers=layers, data_path=DATA_PATH, name="einsum_")
