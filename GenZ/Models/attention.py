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

# TODO: implement Latent attention: https://www.youtube.com/watch?v=0VLAoVGf_74
# https://arxiv.org/pdf/2405.04434
# Attention Mechanism KV Cache per Token (# Element) Capability
# Multi-Head Attention (MHA)    2𝑛ℎ𝑑ℎ𝑙     Strong
# Grouped-Query Attention (GQA) 2𝑛𝑔𝑑ℎ𝑙  Moderate
# Multi-Query Attention (MQA)   2𝑑ℎ𝑙      Weak
# MLA (Ours)                    (𝑑𝑐 +𝑑𝑅ℎ)𝑙 ≈9/2 𝑑ℎ𝑙 Stronger

# Table 1 |Comparison of the KV cache per token among different attention mechanisms.
# 𝑛ℎ denotes the number of attention heads
# 𝑑ℎ denotes the dimension per attention head,
# 𝑙 denotes the number of layers
# 𝑛𝑔 denotes the number of groups in GQA, and
# 𝑑𝑐 and 𝑑𝑅ℎ denote the KV compression dimension and the per-head dimension of the decoupled queries and key in MLA, respectively.
# The amount of KV cache is measured by the number of elements, regardless of the
# storage precision. For DeepSeek-V2, 𝑑𝑐 is set to 4𝑑ℎ and 𝑑𝑅ℎ is set to 𝑑ℎ/2 .
# So, its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA.

    ## [Batch/dp, Seq/sp, Dmodel] * [2, Dmodel, Dq, Hkv/tp] + [Dmodel, Dq, Head/tp]= [Batch/dp, Seq/sp, 3, Dq, Head/tp]
    QKV =           [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if sp > 1:
        seq_all2all = [["Seq A2A", input_sequence_length//sp, D, 1, 1, sp, CollectiveType.All2All, OpType.Sync]]
    else:
        seq_all2all = []

    ## [Batch/dp, Seq, Dq, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Seq/sp, Head/tp]
    logit =         [["Logit",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [["Attend",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Dq, Head/tp,  Dmodel] = [Batch/dp, Seq, Dmodel]
    output =        [["Out Proj", D, input_sequence_length//sp, (per_node_H) * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if sp > 1:
        seq_rs = [["Seq RS", input_sequence_length//sp, D, 1, 1, sp, CollectiveType.ReduceScatter, OpType.Sync]]
    else:
        seq_rs = []

    if tp > 1:
        sync =          [["MHA AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    lora_ops = []
    if model_config.lora_rank > 0:
        S = input_sequence_length // sp
        # LORA for Q
        # GEMM: [op_name, M_output_features, N_sequence_length, K_input_features, Batch, group_factor, ResidencyInfo, OpType.GEMM]
        # LORA_A_Q: Output(S, r), Input(S, D), Weight(D, r)
        lora_ops.append(["LORA_A_Q", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM])
        # LORA_B_Q: Output(S, per_node_H*Dq), Input(S, r), Weight(r, per_node_H*Dq)
        lora_ops.append(["LORA_B_Q", per_node_H*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM])
        # LORA for V
        # LORA_A_V: Output(S, r), Input(S, D), Weight(D, r)
        lora_ops.append(["LORA_A_V", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM])
        # LORA_B_V: Output(S, per_node_Hkv*Dq), Input(S, r), Weight(r, per_node_Hkv*Dq)
        lora_ops.append(["LORA_B_V", per_node_Hkv*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM])

    return QKV + lora_ops + seq_all2all + logit + attend + output + seq_rs + sync

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

    if sp > 1:
        seq_all2all = [["Seq A2A", 1, D, 1, 1, sp, CollectiveType.All2All, OpType.Sync]]
    else:
        seq_all2all = []

    logit_pre =     [["Logit Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [["Attend Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [["Logit Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
    attend_suf =    [["Attend Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
    output =        [["Out Proj",D, 1, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    if sp > 1:
        seq_rs = [["Seq RS", 1, D, 1, 1, sp, CollectiveType.ReduceScatter, OpType.Sync]]
    else:
        seq_rs = []
    if tp > 1:
        sync =          [["MHA AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    lora_ops = []
    if model_config.lora_rank > 0:
        S = 1 # Sequence length for decode is 1
        # LORA for Q
        # GEMM: [op_name, M_output_features, N_sequence_length, K_input_features, Batch, group_factor, ResidencyInfo, OpType.GEMM]
        # LORA_A_Q: Output(S, r), Input(S, D), Weight(D, r)
        lora_ops.append(["LORA_A_Q", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA_B_Q: Output(S, per_node_H*Dq), Input(S, r), Weight(r, per_node_H*Dq)
        lora_ops.append(["LORA_B_Q", per_node_H*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA for V
        # LORA_A_V: Output(S, r), Input(S, D), Weight(D, r)
        lora_ops.append(["LORA_A_V", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA_B_V: Output(S, per_node_Hkv*Dq), Input(S, r), Weight(r, per_node_Hkv*Dq)
        lora_ops.append(["LORA_B_V", per_node_Hkv*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])

    return query + lora_ops + seq_all2all + logit_pre + logit_suf + attend_pre + attend_suf + output + seq_rs + sync



def mha_flash_attention_chunked(model_config:ModelConfig, parallelism_config:ParallelismConfig,
                                chunk_size: int, prefill_kv_sizes: list[int,int], decode_kv_sizes: list[int]):
    '''
        Generates a list of operators for multi-head attention (MHA) with flash attention,
        chunked processing, and parallelism configurations.
        Args:
            model_config (ModelConfig): Configuration object containing model parameters such as
                                        number of attention heads, key-value heads, hidden size, and head dimension.
            parallelism_config (ParallelismConfig): Configuration object containing parallelism parameters
                                                    such as tensor parallelism, expert parallelism, sequence parallelism, and data parallelism.
            chunk_size (int): Maximum chunk size of the values to be processed.
            prefill_kv_sizes (int): List of sizes of the prefill
                                    First value of tuple is the tokens processed till now and second value is the tokens processed in the current chunk.
            decode_kv_sizes (list[int]): List of sizes for the key-value pairs during the decode stage.

            The call to this function should handle the prefill_kv_sizes calculation.
        Returns:
            list: A list of layers with their respective configurations for the MHA with flash attention.

    '''
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)


    layers = []
    query =      [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += query

    if model_config.lora_rank > 0:
        S = chunk_size
        # LORA for Q
        # GEMM: [op_name, M_output_features, N_sequence_length, K_input_features, Batch, group_factor, ResidencyInfo, OpType.GEMM]
        # LORA_A_Q: Output(S, r), Input(S, D), Weight(D, r)
        layers.append(["LORA_A_Q", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA_B_Q: Output(S, per_node_H*Dq), Input(S, r), Weight(r, per_node_H*Dq)
        layers.append(["LORA_B_Q", per_node_H*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA for V
        # LORA_A_V: Output(S, r), Input(S, D), Weight(D, r)
        layers.append(["LORA_A_V", model_config.lora_rank, S, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])
        # LORA_B_V: Output(S, per_node_Hkv*Dq), Input(S, r), Weight(r, per_node_Hkv*Dq)
        layers.append(["LORA_B_V", per_node_Hkv*Dq, S, model_config.lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM])

    if sp > 1:
        seq_all2all = [["Seq A2A", chunk_size, D, 1, 1, sp, CollectiveType.All2All, OpType.Sync]]
        layers += seq_all2all

    ## Prefill LA layers
    for kv_size in prefill_kv_sizes:
        layers +=    [["Logit Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]
        layers +=    [["Attend Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]

    ## Decode LA layers
    for kv_size in decode_kv_sizes:
        if isinstance(kv_size, tuple) and len(kv_size) == 4:
            prefill_num_beams = kv_size[0]
            decode_num_beams = kv_size[1]
            prefill_context = kv_size[2]//sp
            decode_context = kv_size[3]
            # layers +=    [["Logit Dec",per_node_H, 1, kv_size[1], Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            # layers +=    [["Attend Dec",per_node_H, 1, kv_size[1], Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
            layers +=     [["Logit Pre",(prefill_num_beams*per_node_H), 1, prefill_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Pre",(prefill_num_beams*per_node_H), 1, prefill_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
            layers +=     [["Logit Suf",(decode_num_beams*per_node_H), 1, decode_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Suf",(decode_num_beams*per_node_H), 1, decode_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
        elif isinstance(kv_size, tuple) and len(kv_size) == 2:
            num_batches = kv_size[0]
            past_context = kv_size[1]
            layers +=     [["Logit Dec",num_batches*per_node_H, 1, past_context, Dq, num_batches*per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Dec",num_batches*per_node_H, 1, past_context, Dq, num_batches*per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
        else:
            layers +=     [["Logit Dec",per_node_H, 1, kv_size, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Dec",per_node_H, 1, kv_size, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]


    layers +=        [["Out Proj",D, chunk_size, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if sp > 1:
        layers += [["Seq RS", chunk_size, D, 1, 1, sp, CollectiveType.ReduceScatter, OpType.Sync]]
    if tp > 1:
        layers +=          [["MHA AR",chunk_size, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers

