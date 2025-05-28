from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_decode_model
from math import ceil

unit = Unit()

def decode_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    output_tokens = 0,   Bb = 4 ,           ## Only for Decode
    system_name = 'A100_40GB_GPU', system_eff = 1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy='GenZ', network_config=None,
    parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
    model_offload = False, ceff = None, meff = None):

    if pipeline_parallel > 1:
        ub = max(batch_size // pipeline_parallel, 1)
        num_micro_batches = batch_size // ub
        if batch_size < pipeline_parallel:
            warnings.warn(f"Batch size is divided into micro batches for pipeline parallel, micro batch size:{ub}, consider increasing batch size")
    else:
        ub = batch_size

    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    system = get_inference_system(system_name = system_name, bits = bits, ceff=system_eff , meff=system_eff,
                                network_config=network_config, 
                                collective_strategy=collective_strategy, 
                                parallelism_heirarchy=parallelism_heirarchy )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    # if is_moe:
    # Obtain ModelConfig to access LORA parameters
    from GenZ.Models.get_language_model import get_configs
    model_config_obj = get_configs(model)

    model_decode_filename = create_full_decode_model(name=model_config_obj, # Pass the object
                                            input_sequence_length=input_tokens,
                                            output_gen_tokens = output_tokens ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_decode_filename, system=system, batch_size= ub*Bb, intermediate_on_chip=True , beam_merge= (Bb > 1), beam_size= Bb, model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)

    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB
    unused_weights = summary_table[f'Unused Weights ({unit.unit_mem})'].values[0]      ## In MB

    # Calculate LORA weights if lora_rank > 0
    lora_weights_mb = 0
    if model_config_obj.lora_rank > 0:
        bytes_per_parameter = 0
        if bits == 'bf16' or bits == 'fp16':
            bytes_per_parameter = 2
        elif bits == 'fp32':
            bytes_per_parameter = 4
        elif bits == 'int8':
            bytes_per_parameter = 1
        else:
            warnings.warn(f"Unsupported bits type {bits} for LORA weight calculation. Assuming 2 bytes (bf16/fp16).")
            bytes_per_parameter = 2

        # Q LORA matrices
        q_lora_a_size = model_config_obj.hidden_size * model_config_obj.lora_rank * bytes_per_parameter
        # Note: num_attention_heads is the total before TP sharding.
        # The LORA B matrix output dimension should match the sharded Q projection output dimension.
        # However, the problem description implies full dimensions for LORA B: (lora_rank, head_dim * num_heads)
        # Let's assume head_dim * num_attention_heads is the intended full output dimension before sharding for Q
        q_lora_b_size = model_config_obj.lora_rank * (model_config_obj.head_dim * model_config_obj.num_attention_heads) * bytes_per_parameter
        
        # V LORA matrices
        v_lora_a_size = model_config_obj.hidden_size * model_config_obj.lora_rank * bytes_per_parameter
        # Similarly for V, using num_key_value_heads
        v_lora_b_size = model_config_obj.lora_rank * (model_config_obj.head_dim * model_config_obj.num_key_value_heads) * bytes_per_parameter
        
        total_lora_weights_one_layer_bytes = q_lora_a_size + q_lora_b_size + v_lora_a_size + v_lora_b_size
        total_lora_weights_all_layers_bytes = model_config_obj.num_decoder_layers * total_lora_weights_one_layer_bytes
        
        lora_weights_mb = total_lora_weights_all_layers_bytes / (1024 * 1024) # Convert bytes to MB

    model_weights += lora_weights_mb

    total_memory_req = model_weights + kv_cache
    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if  per_chip_memory < total_memory_req/pipeline_parallel:
        if model_offload:
            system = get_offload_system(system=system, total_memory_req = total_memory_req/pipeline_parallel , debug=debug)
            warnings.warn(f"Some Parameter offloaded, effective Memory BW:{unit.raw_to_unit(system.offchip_mem_bw, type='BW')} ")
            is_offloaded = True
        elif model_profilling:
            warnings.warn(f"All params would not fit on chip. System Memory Cap:{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, KV Cache:{kv_cache/1024} ")
        else:
            raise ValueError(f"All params would not fit on chip. System Memory Cap:{per_chip_memory/1024} GB , Weights : {model_weights/1024} GB, KV Cache:{kv_cache/1024}. \n System:{system_name}")

    ## for tensor shareding per layer.
    assert pipeline_parallel >= 1, "Pipeline parallel must be >= 1"
    assert tensor_parallel >= 1, f"Tensor parallel must be >= 1, {tensor_parallel}"

    if model_profilling:
        return model_df, summary_table

    ##################################################################################################
    ### Token generation time
    ##################################################################################################
    # model_decode = create_full_decode_model(name=model,
    #                                         input_sequence_length=input_tokens,
    #                                         output_gen_tokens = output_tokens ,
    #                                         tensor_parallel=tensor_parallel,
    #                                         pipeline_parallel=pipeline_parallel,
    #                                         expert_parallel=expert_parallel)

    model_df = get_model_df(model_decode_filename, system, unit, ub*Bb,  intermediate_on_chip=True , beam_merge= (Bb > 1), beam_size= Bb) # Use model_decode_filename
    summary_table = get_summary_table(model_df, unit)

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)
    decode_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]      # Latency in msec

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    # if pipeline_parallel > 1:
    #     micro_batch_latency = decode_latency
    #     ## If the N micro batches, then the total latency is (N-1)*stage latency + initial_latency
    #     ## We make the assumption that the pipeline is balanced and the latency is same for all stages
    #     total_latency = ((num_micro_batches-1) * (decode_latency / pipeline_parallel)) + micro_batch_latency
    #     thrpt = 1000 * batch_size / total_latency
    # else:
    thrpt = 1000 * batch_size / decode_latency


    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]                ## In milliseconds
    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]                    ## In milliseconds
    total_communication_delay = summary_table[f'Comm Latency ({unit.unit_time})'].values[0]    ## In milliseconds
    total_time = linear_time + attn_time + total_communication_delay
    # runtime_breakdown = [linear_time, attn_time, total_communication_delay]
    runtime_breakdown = get_runtime_breakdown(model_df)
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=decode_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                )
