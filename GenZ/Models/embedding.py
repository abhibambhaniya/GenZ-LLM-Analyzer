from GenZ.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from GenZ.parallelism import ParallelismConfig
import warnings

def input_embedding(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel

    D = model_config.hidden_size
    V = model_config.vocab_size


    emb =   [["embeddings", D, max(1,input_sequence_length//sp), V//tp, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["Emb_AR", max(1,input_sequence_length//sp), D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return emb + sync

def output_embedding(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel

    D = model_config.hidden_size
    V = model_config.vocab_size


    emb =   [["classifier", V//tp, max(1,input_sequence_length//sp), D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["classifier_AG",max(1,input_sequence_length//sp), V, 1, 1, tp, CollectiveType.AllGather, OpType.Sync]]
    else:
        sync = []

    return emb + sync