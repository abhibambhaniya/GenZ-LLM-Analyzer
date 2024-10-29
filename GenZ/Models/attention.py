from GenZ.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from GenZ.parallelism import ParallelismConfig
from math import ceil

def mha_flash_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    ## [Batch/dp, Seq/sp, Dmodel] * [2, Dmodel, Dq, Hkv/tp] + [Dmodel, Dq, Head/tp]= [Batch/dp, Seq/sp, 3, Dq, Head/tp]
    QKV =           [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Seq/sp, Head/tp]
    logit =         [["Logit",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [["Attend",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Dq, Head/tp,  Dmodel] = [Batch/dp, Seq, Dmodel]
    output =        [["Out Proj", D, input_sequence_length//sp, (per_node_H) * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["MHA AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []
    return QKV + logit + attend + output + sync

def mha_flash_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    query =         [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    logit_pre =     [["Logit Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [["Attend Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [["Logit Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
    attend_suf =    [["Attend Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
    output =        [["Out Proj",D, 1, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        sync =          [["MHA AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return query + logit_pre + logit_suf + attend_pre + attend_suf + output + sync

