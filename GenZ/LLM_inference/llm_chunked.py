from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_chunked_model
from math import ceil

unit = Unit()

def chunked_moddeling(model = 'BERT',
    prefill_kv_sizes = [(1024, 511)],                 # [(prefill_past_kv, num_prefill)],
    decode_kv_sizes =  [1600, 1601, 1602],            # [decode_past_kv]*num_decodes,
    system_name = 'A100_40GB_GPU', system_eff = 1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy='GenZ', network_config=None,
    parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
    model_offload = False, ceff = None, meff = None):

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
    model_chunked = create_full_chunked_model(name=model,
                                            prefill_kv_sizes= prefill_kv_sizes,
                                            decode_kv_sizes = decode_kv_sizes,
                                            tensor_parallel = tensor_parallel, pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_chunked, system=system, batch_size= 1, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)

    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB
    unused_weights = summary_table[f'Unused Weights ({unit.unit_mem})'].values[0]      ## In MB

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
    # model_chunked = create_full_chunked_model(name=model,
    #                                         prefill_kv_sizes= prefill_kv_sizes,
    #                                         decode_kv_sizes = decode_kv_sizes,
    #                                         tensor_parallel = tensor_parallel, pipeline_parallel = pipeline_parallel,
    #                                         expert_parallel=expert_parallel)

    model_df = get_model_df(model_chunked, system, unit, 1,  intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)
    chunked_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]      # Latency in msec

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s

    thrpt = 1000 * (sum([i[1] for i in prefill_kv_sizes]) + len(decode_kv_sizes)) / chunked_latency

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
                        Latency=chunked_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                )