from GenZ.Models import ModelConfig, ResidencyInfo, OpType, parse_einsum_expression, CollectiveType
from GenZ.parallelism import ParallelismConfig
import numpy as np
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
    logL = round(np.log10(L))
    Layers = []
    ## A and intermediate tensors on chip
    deltaA = [["deltaA",'bdl,dn->bdln',parse_einsum_expression('bdl,dn->bdln', ('b',F,L), (F,S), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.BC_onchip, OpType.EINSUM]]
    ## output off chip
    deltaB = [["deltaB",'bdl,bls->bdls',parse_einsum_expression('bdl,bls->bdls', ('b',F,L), ('b',L,S), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.C_onchip, OpType.EINSUM]]
    ## U and output on-chip
    deltaB_u = [["deltaBu",'bdls,bdl->bdls',parse_einsum_expression('bdls,bdl->bdls', ('b',F,L,S), ('b',F,L), ('b',F,L,S)), 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]
    Layers += deltaA + deltaB + deltaB_u
    # for _ in range(L):
    Layers += [["x calc",'lbfs,lbfs->lbfs',parse_einsum_expression('xbfs,lbfs->xbfs', (logL,'b',F,S), (L,'b',F,S), (logL,'b',F,S)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]]
    Layers += [["y calc",'lbfs,lbs->lbf',parse_einsum_expression('xbfs,lbs->xbf', (logL,'b',F,S), (L,'b',S), (logL,'b',F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]]
    Layers += [["D addition",'blf,blf->blf',parse_einsum_expression('blf,blf->blf', ('b',L,F), ('b',L,F), ('b',L,F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]] 
    Layers += [["out mult z",'blf,blf->blf',parse_einsum_expression('blf,blf->blf', ('b',L,F), ('b',L,F), ('b',L,F)), 1, 1, 1, ResidencyInfo.All_onchip, OpType.EINSUM]] 

    return Layers

def mamba_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    L  = input_sequence_length ## input Seq Len

    D = round(model_config.hidden_size/tp)

    ## Mamba parameters
    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}'

    in_proj =      [["Inproj",2*F, L, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [["Conv",F, F, L, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [["BC proj",R+2*S, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [["xt proj",F, L, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ssn =   mamda_ssn_slow(model_config, parallelism_config, input_sequence_length)

    output =       [["Out proj",D, L, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["Mamba AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return in_proj + conv_1d + dbc_proj + xt_proj + ssn + output + sync


def mamba_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig,
                input_sequence_length:int):

    L  = input_sequence_length ## input Seq Len

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    ## Mamba parameters
    D = round(model_config.hidden_size/tp)

    S = model_config.mamba_d_state
    C = model_config.mamba_d_conv
    F = D * model_config.mamba_expand
    R = model_config.mamba_dt_rank

    in_proj =      [["Inproj",2*F, 1, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]        ## BLD * D2F = BL2F
    conv_1d =      [["Conv", F, F, 1, C, 1, ResidencyInfo.All_offchip, OpType.CONV1D]]         ## BLF conv FC -> BLF
    dbc_proj =     [["BC proj",R+2*S, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    xt_proj =      [["xt proj",F, 1, R, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    ssn =   mamda_ssn_slow(model_config, parallelism_config, 1)
    output =       [["Out proj",D, 1, F, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    if tp > 1:
        sync =          [["Mamba AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return in_proj + conv_1d + dbc_proj + xt_proj + ssn + output + sync