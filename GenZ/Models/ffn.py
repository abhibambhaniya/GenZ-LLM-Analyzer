from GenZ.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from GenZ.parallelism import ParallelismConfig
import warnings

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

    if tp > 1:
        sync =          [[input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return ffup + ffdown + sync

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
    if tp > 1:
        sync =          [[1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers + sync