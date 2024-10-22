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
    EINSUM = 12

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
    logit =         [[H//tp, input_sequence_length, input_sequence_length//sp, Dq, max(Hkv//tp,1), ResidencyInfo.C_onchip, OpType.Logit]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [[H//tp, input_sequence_length, input_sequence_length//sp, Dq, max(Hkv//tp,1), ResidencyInfo.A_onchip, OpType.Attend]]
    
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
    logit_pre =     [[H//tp, 1, input_sequence_length//sp, Dq, Hkv//tp, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [[H//tp, 1, input_sequence_length//sp, Dq, Hkv//tp, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [[H//tp, 1, output_gen_tokens, Dq, Hkv//tp, ResidencyInfo.AC_onchip, OpType.Logit]]
    attend_suf =    [[H//tp, 1, output_gen_tokens, Dq, Hkv//tp, ResidencyInfo.AC_onchip, OpType.Attend]]
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

def mamda_ssn_slow(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    L = input_sequence_length ## input Seq Len
    ## Mamba parameters
    D = model_config.hidden_size
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    """
    u: r(B F L)
    delta: r(B F L)
    A: r(F S)
    B: r(B S L)
    C: r(B S L)
    D: r(F)
    z: r(B F L)
    delta_bias: r(F), fp32

    out: r(B F L)
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])    
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y + u * rearrange(D, "d -> d 1")
    out = out * F.silu(z)
    """

    Layers = []
    ## A and intermediate tensors on chip
    deltaA = [['bdl,dn->bdln',parse_einsum_expression('bdl,dn->bdln', ('b',F,L), (F,S), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.BC_onchip, OpType.EINSUM]]
    ## output off chip
    deltaB = [['bdl,bls->bdls',parse_einsum_expression('bdl,bls->bdls', ('b',F,L), ('b',L,S), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.C_onchip, OpType.EINSUM]]
    ## U and output on-chip
    deltaB_u = [['bdls,bdl->bdls',parse_einsum_expression('bdls,bdl->bdls', ('b',F,L,S), ('b',F,L), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]
    Layers += deltaA + deltaB + deltaB_u
    # for _ in range(L):
    Layers += [['lbfs,lbfs->lbfs',parse_einsum_expression('lbfs,lbfs->lbfs', (L,'b',F,S), (L,'b',F,S), (L,'b',F,S)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]]
    Layers += [['lbfs,lbs->lbf',parse_einsum_expression('lbfs,lbs->lbf', (L,'b',F,S), (L,'b',S), (L,'b',F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]]
    Layers += [['blf,blf->blf',parse_einsum_expression('blf,blf->blf', ('b',L,F), ('b',L,F), ('b',L,F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]] 
    Layers += [['blf,blf->blf',parse_einsum_expression('blf,blf->blf', ('b',L,F), ('b',L,F), ('b',L,F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]] 

    return Layers 

def mamba_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    L  = input_sequence_length ## input Seq Len

    D = model_config.hidden_size
    Df = model_config.intermediate_size

    ## Mamba parameters
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}'
    Df = max(Df//tp,1)

    in_proj =      [[2*F, L, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [[F, F, L, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [[R+2*S, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [[F, L, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ssn =   mamda_ssn_slow(model_config, parallelism_config, input_sequence_length)

    output =       [[D, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    return in_proj + conv_1d + dbc_proj + xt_proj + ssn + output


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


def mamba_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig,
                input_sequence_length:int):

    L  = input_sequence_length ## input Seq Len

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    ## Mamba parameters
    D = model_config.hidden_size
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    in_proj =      [[2*F, 1, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [[F, F, 1, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [[R+2*S, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [[F, 1, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ssn =   mamda_ssn_slow(model_config, parallelism_config, 1)
    output =       [[D, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    return in_proj + conv_1d + dbc_proj + xt_proj + ssn + output

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

def parse_einsum_expression(expression, *tensors):
    einsum_vars = {}
    input_subscripts, output_subscript = expression.split('->')
    input_subscripts = input_subscripts.split(',')

    for tensor, subscripts in zip(tensors, input_subscripts):
        for dim, subscript in zip(tensor, subscripts):
            if subscript not in einsum_vars:
                einsum_vars[subscript] = dim
            elif einsum_vars[subscript] != dim:
                raise ValueError(f"Dimension mismatch for subscript: {subscript}, Got: {dim}, Expected: {einsum_vars[subscript]}")

    for subscript in output_subscript:
        if subscript not in einsum_vars:
            einsum_vars[subscript] = None

    return einsum_vars

def einsum_test(equation=None, einsum_vars=None):

    if equation is None:
        A = (2, 3, 4)
        B = (2, 4, 5)
        C = (5, 6)
        equation = 'ijk,ikl,lm->ijm'
        einsum_vars = parse_einsum_expression(equation, A, B, C)

    layers = [[equation, einsum_vars, 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]

    return save_layers(layers=layers, data_path=DATA_PATH, name="einsum_")
