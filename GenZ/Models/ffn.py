from GenZ.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from GenZ.parallelism import ParallelismConfig
import warnings
from math import ceil

def ffn_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):

    tp = parallelism_config.tensor_parallel
    ep = parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi

    E = model_config.num_experts
    K = model_config.expert_top_k
    Df = max(ceil(Df/tp),1)
    moe_layer = (E > 1)

    if E == 1 and ep > 1:
        warnings.warn(f"For dense model, expert parallelism:{ep} will be treated as model parallel")

    assert E % ep == 0, f"Number of experts:{E} must be divisible by expert parallelism:{ep}"

    assert E >= ep, f"Number of experts:{E} must be less than expert parallelism:{ep}"

    layers = []
    if moe_layer:
        router = [["Gate",E, input_sequence_length//sp, D//tp, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        layers += router
        if tp > 1:
            router_AR = [["Gate AR",input_sequence_length//sp, E, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
            layers += router_AR
        num_tokens_per_expert = (input_sequence_length//sp) * K // E
        if ep > 1:
            # Total Size=Batch Size×Tokens per Batch×Hidden Dimension×Number of Experts per Token
            dispatch_all2all = [["Dispatch A2A",input_sequence_length//sp, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += dispatch_all2all
        ffup =   [["up+gate",(E//ep)*Df*fi, num_tokens_per_expert, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [["down",D, num_tokens_per_expert, (E//ep)*Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

        layers += ffup + ffdown
        if ep > 1:
            collect_all2all = [["Collect A2A",input_sequence_length//sp, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += collect_all2all
    else:
        ffup =   [["up+gate",Df*fi, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [["down",D, input_sequence_length//sp, Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        layers += ffup + ffdown

    if tp > 1:
        sync =          [["FFN AR",input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers + sync

def ffn_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig):
    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi

    tp = parallelism_config.tensor_parallel
    ep = parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    E = model_config.num_experts
    K = model_config.expert_top_k
    Df = max(ceil(Df/tp),1)
    moe_layer = (E > 1)

    if E == 1 and ep > 1:
        warnings.warn(f"For dense model, expert parallelism:{ep} will be treated as model parallel")

    assert E % ep == 0, f"Number of experts:{E} must be divisible by expert parallelism:{ep}"
    assert E >= ep, f"Number of experts:{E} must be less than expert parallelism:{ep}"


    layers = []
    if moe_layer:
        router = [["Gate",E, 1, D//tp, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        layers += router
        if tp > 1:
            router_AR = [["Gate AR",1, E, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
            layers += router_AR
        if ep > 1:
            # Total Size=Batch Size×Tokens per Batch×Hidden Dimension×Number of Experts per Token
            dispatch_all2all = [["Dispatch A2A",1, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += dispatch_all2all

        ## TODO: Define a function to calculate the number of activated experts
        A = K

        npus_activated = max(1,ceil(A/ep))
        ## Understanding load imbalance among experts
        # Lets' say we have 4 experts and 2 experts per token
        # E = 4, k = 2 ,ep = 2, Activated = 2
        # Best case: activated experts are distributed among EP
        #   Then FF: Df*fi*max(1, ceil(A/ep) )
        # Worst case: all activated experts are in the same EP
        #   Then FF: Df*fi*min(A, ceil(E/ep))
        #   E =  16, k = 2, ep = 4, A = 3
        #   Best case: max(1, 3/4) = 1 expert per chip
        #   Worst case: min(4, 16/4) = 4 expert per chip
        #
        #   E =  16, k = 2, ep = 4, A = 5
        #   Best case: max(1, 5//4) = 2 expert per chip
        #   Worst case: min(5, 16//4) = 4 expert per chip

        ## Activated experts are distributed among EP
        ffup =           [["up+gate",npus_activated*Df*fi, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]    ## Df is already divided
        ffdown =           [["down",D, 1, npus_activated*Df, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

        ## These are unused layers but kept just for weights calculation
        ffup_unused =   [["up+gate",(E-A)*Df*fi, 0, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown_unused =   [["down",D, 0, (E-A)*Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

        layers += ffup + ffdown + ffup_unused + ffdown_unused
        if ep > 1:
            collect_all2all = [["Collect A2A",1, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += collect_all2all
    else:
        ffup =           [["up+gate",Df*fi, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]    ## Df is already divided
        ffdown =           [["down",D, 1, Df, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
        layers += ffup + ffdown

    if tp > 1:
        sync =          [["FFN AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers + sync