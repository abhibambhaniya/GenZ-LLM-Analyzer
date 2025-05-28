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
            router_AR = [["Gate AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
            layers += router_AR
        if ep > 1:
            # Total Size=Batch Size×Tokens per Batch×Hidden Dimension×Number of Experts per Token
            dispatch_all2all = [["Dispatch A2A",1, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += dispatch_all2all

        ## TODO: Define a function to calculate the number of activated experts
        A = K

        experts_activated_per_chip = max(1,ceil(A/ep))
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
        ffup =           [["up+gate",experts_activated_per_chip*Df*fi, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]    ## Df is already divided
        ffdown =           [["down",D, 1, experts_activated_per_chip*Df, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

        ## These are unused layers but kept just for weights calculation
        unsed_experts_per_chip = ceil(E/ep) - experts_activated_per_chip
        ffup_unused =   [["up+gate",unsed_experts_per_chip*Df*fi, 0, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown_unused =   [["down",D, 0, unsed_experts_per_chip*Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

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



def deepseek_ffn_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):

    tp = parallelism_config.tensor_parallel
    ep = parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    D = model_config.hidden_size
    fi = model_config.num_ffi

    E = model_config.num_experts
    K = model_config.expert_top_k

    n_shared_experts = model_config.n_shared_experts

    moe_layer = (E > 1)

    if E == 1 and ep > 1:
        warnings.warn(f"For dense model, expert parallelism:{ep} will be treated as model parallel")

    assert E % ep == 0, f"Number of experts:{E} must be divisible by expert parallelism:{ep}"

    assert E >= ep, f"Number of experts:{E} must be less than expert parallelism:{ep}"

    layers = []
    if moe_layer:
        Df_moe = model_config.moe_intermediate_size
        Df_moe = max(ceil(Df_moe/tp),1)
        router = [["Gate",E, input_sequence_length//sp, D//tp, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        layers += router
        if tp > 1:
            router_AR = [["Gate AR",input_sequence_length//sp, E, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
            layers += router_AR

        ## TODO: Define a function to calculate the number of activated experts
        '''
        Load Imbalance Factor = Highest Number of Token to Single Expert / Equal Distribution of tokens to expert
        - A single expert can recieve at most input_sequence_length tokens and min 0 tokens
        - Minimum number of experts activated is K.
        - With equal distribution, each expert will get K*(input_sequence_length)/E
        - For perfect distribution, load imbalance factor = 1
        - For worst case, load imbalance factor = input_sequence_length / (K*input_sequence_length/E) = E/K

        All2All traffic with load imbalance factor
        - All2All traffic total data = input_sequence_length * K * D
        - Perfect balance All2All traffic per chip = ((input_sequence_length * K * D /ep)
        - Worst case All2All traffic in a single chip = ((input_sequence_length * min(K, E/ep) * D )

        Expert FFN time
        Best case:
            # if K <= EP:
                1 expert activated per EP.
                Tokens per expert = input_sequence_length
            # if K > EP:
                more than 1 expert activated per EP.
                Tokens per expert = input_sequence_length * ceil( K / EP)
        Worst case:
            
        '''
        A = min( K * input_sequence_length, E)
        experts_activated_per_chip = max(1,ceil(A/ep))
        num_tokens_per_expert = (input_sequence_length//sp) * K // A if A > 0 else 0

        if ep > 1:
            # Total Size=Batch Size×Tokens per Batch×Hidden Dimension×Number of Experts per Token
            dispatch_all2all = [["Dispatch A2A",input_sequence_length//sp, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += dispatch_all2all
        ffup =   [["up+gate", experts_activated_per_chip*Df_moe*fi, num_tokens_per_expert, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [["down",D, num_tokens_per_expert, experts_activated_per_chip*Df_moe, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

        layers += ffup + ffdown

        ## These are unused layers but kept just for weights calculation
        unsed_experts_per_chip = ceil(E/ep) - experts_activated_per_chip
        if unsed_experts_per_chip:
            ffup_unused =   [["up+gate",unsed_experts_per_chip*Df_moe*fi, 0, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
            ffdown_unused =   [["down",D, 0, unsed_experts_per_chip*Df_moe, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
            layers += ffup_unused + ffdown_unused
        if ep > 1:
            collect_all2all = [["Collect A2A",input_sequence_length//sp, K*D, 1, 1, ep, CollectiveType.All2All, OpType.Sync]]
            layers += collect_all2all

        if n_shared_experts > 0:
            Df = model_config.shared_expert_intermediate_size
            Df = max(ceil(Df/tp),1)

            ffup =   [["shared up+gate",Df*fi*n_shared_experts, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
            ffdown = [["shared down",D, input_sequence_length//sp, Df*n_shared_experts, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
            layers += ffup + ffdown

    else:
        Df = model_config.intermediate_size
        Df = max(ceil(Df/tp),1)
        ffup =   [["up+gate",Df*fi, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        ffdown = [["down",D, input_sequence_length//sp, Df, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        layers += ffup + ffdown

    if tp > 1:
        sync =          [["FFN AR",input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers + sync