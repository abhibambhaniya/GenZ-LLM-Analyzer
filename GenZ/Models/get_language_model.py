import pandas as pd
import os
from math import ceil
import numpy as np
from datetime import datetime
from enum import IntEnum
import warnings
from GenZ.parallelism import ParallelismConfig

from GenZ.Models.default_models import ModelConfig, MODEL_DICT

class OpType(IntEnum):
    FC = 0
    CONV2D = 1
    DWCONV = 2
    GEMM = 3
    Logit = 4
    Attend = 5
    Sync = 6
    Logit_MQA = 7
    Attend_MQA = 8
    Logit_BM_PREFILL = 9
    Attend_BM_PREFILL = 10
    CONV1D = 11

class ResidencyInfo(IntEnum):
    All_offchip = 0
    A_onchip = 1
    B_onchip = 2
    C_onchip = 3
    AB_onchip = 4
    AC_onchip = 5
    BC_onchip = 6
    All_onchip = 7


def get_configs(name) -> ModelConfig:
    name = name.lower()

    if model := MODEL_DICT.get(name):
        model_config = model
    else:
        ## If unknown name, then giving parameters of BERT
        print("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

    return model_config

def mha_flash_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    ## [Batch/dp, Seq/sp, Dmodel] * [2, Dmodel, Dq, Hkv/tp] + [Dmodel, Dq, Head/tp]= [Batch/dp, Seq/sp, 3, Dq, Head/tp]
    QKV =           [[(H*Dq + 2*Hkv*Dq)//tp, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Seq/sp, Head/tp]
    logit =         [[H//tp, input_sequence_length, input_sequence_length//sp, Dq, max(Hkv//tp,1), ResidencyInfo.All_offchip, OpType.Logit]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [[H//tp, input_sequence_length, input_sequence_length//sp, Dq, max(Hkv//tp,1), ResidencyInfo.All_offchip, OpType.Attend]]
    
    ## [Batch/dp, Seq, Dq, Head/tp] * [Dq, Head/tp,  Dmodel] = [Batch/dp, Seq, Dmodel] 
    output =        [[D, input_sequence_length//sp, (H//tp) * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    return QKV + logit + attend + output

def mha_flash_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    query =         [[(H*Dq + 2*Hkv*Dq)//tp, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    logit_pre =     [[H//tp, 1, input_sequence_length//sp, Dq, Hkv//tp, ResidencyInfo.All_offchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [[H//tp, 1, input_sequence_length//sp, Dq, Hkv//tp, ResidencyInfo.All_offchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [[H//tp, 1, output_gen_tokens, Dq, Hkv//tp, ResidencyInfo.All_offchip, OpType.Logit]]
    attend_suf =    [[H//tp, 1, output_gen_tokens, Dq, Hkv//tp, ResidencyInfo.All_offchip, OpType.Attend]]
    output =        [[D, 1, (H//tp) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    return query + logit_pre + logit_suf + attend_pre + attend_suf + output

def ffn_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):

    tp = parallelism_config.tensor_parallel
    ep = parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Df = max(Df//tp,1)

    if E == 1 and ep > 1:
        warnings.warn(f"For dense model, expert parallelism:{ep} will be treated as model parallel")
    
    assert E % ep == 0, f"Number of experts:{E} must be divisible by expert parallelism:{ep}"
    
    if moe_layer_freq:
        num_tokens_per_expert = (input_sequence_length//sp) * K // E
        ffup =   [[(E//ep)*Df*fi, num_tokens_per_expert, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [[D, num_tokens_per_expert, (E//ep)*Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    else:
        ffup =   [[Df*fi, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [[D, input_sequence_length//sp, Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    return ffup + ffdown

def ffn_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig):
    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi

    tp = parallelism_config.tensor_parallel
    ep = parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Df = max(Df//tp,1)

    if E == 1 and ep > 1:
        warnings.warn(f"For dense model, expert parallelism:{ep} will be treated as model parallel")

    assert E % ep == 0, f"Number of experts:{E} must be divisible by expert parallelism:{ep}"
    
    ffup =           [[K*Df*fi, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]    ## Df is already divided
    ffdown =           [[D, 1, K*Df, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    ffup_unused =   [[(E-K)*Df*fi, 0, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ffdown_unused =   [[D, 0, (E-K)*Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    layers = []
    layers += (ffup + ffup_unused) if moe_layer_freq  else ffup
    layers += (ffdown + ffdown_unused)  if moe_layer_freq  else ffdown

    return layers

def save_layers(layers:str, data_path:str, name:str):
    model_path = os.path.join(data_path,"model")
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    file_name = name.replace("/", "_") + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") +'.csv'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df.to_csv(os.path.join(model_path, file_name),  header=True, index=None)
    return file_name


DATA_PATH = "/tmp/genz/data/"

def create_inference_moe_prefix_model(input_sequence_length, name='BERT', data_path=DATA_PATH,
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

def create_inference_moe_decode_model(input_sequence_length, name='BERT', data_path=DATA_PATH,
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

def create_inference_mamba_prefix_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)

    L  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    ## Mamba parameters
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}'
    Df = max(Df//tensor_parallel,1)

    layers = []

    in_proj =      [[2*F, L, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [[F, F, L, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [[R+2*S, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [[F, L, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ## TODO:    SSN_kernel
    output =       [[D, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    layers = in_proj + conv_1d + dbc_proj + xt_proj + output

    return save_layers(layers=layers, data_path=data_path, name=name+"_prefill_")

def create_inference_mamba_decode_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)

    L  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    ## Mamba parameters
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}'
    Df = max(Df//tensor_parallel,1)

    layers = []

    in_proj =      [[2*F, 1, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [[F, F, 1, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [[R+2*S, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [[F, 1, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ## TODO:    SSN_kernel
    output =       [[D, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    layers = in_proj + conv_1d + dbc_proj + xt_proj + output

    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")