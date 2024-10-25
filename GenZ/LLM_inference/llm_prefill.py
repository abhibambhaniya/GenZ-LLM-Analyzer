from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import get_configs, create_full_prefill_model

unit = Unit()

def prefill_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    system_name = 'A100_40GB_GPU', system_eff=1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1, return_model_df=False,
    model_offload = False):

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

    system = get_inference_system(system_name = system_name, bits = bits, ceff=system_eff, meff=system_eff)

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_prefill = create_full_prefill_model(input_sequence_length=input_tokens, name=model,
                                                    tensor_parallel=tensor_parallel)


    model_df = get_model_df(model_prefill, system=system, batch_size = ub, intermediate_on_chip=True , model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)

    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB

    total_memory_req = model_weights + kv_cache
    num_nodes = pipeline_parallel * tensor_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if  per_chip_memory*num_nodes< total_memory_req:
        if model_offload:
            system = get_offload_system(system=system, total_memory_req = total_memory_req , debug=debug)
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
    ### Prefill generation time
    ##################################################################################################
    model_prefill = create_full_prefill_model(input_sequence_length=input_tokens, name=model,
                                        tensor_parallel=tensor_parallel)
    model_df = get_model_df(model_prefill, system, unit, ub, intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)
    prefill_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if return_model_df:
        return model_df, summary_table
    if debug:
        display_df(model_df)
        display(summary_table)

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    if pipeline_parallel > 1:
        micro_batch_latency = prefill_latency
        ## If the N micro batches, then the total latency is (N-1)*stage latency + initial_latency
        ## We make the assumption that the pipeline is balanced and the latency is same for all stages
        total_latency = ((num_micro_batches-1) * (prefill_latency / pipeline_parallel)) + micro_batch_latency
        thrpt = 1000 * batch_size / total_latency
    else:
        thrpt = 1000 * batch_size / prefill_latency

    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]
    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]
    total_communication_delay = summary_table[f'Comm Latency ({unit.unit_time})'].values[0]
    runtime_breakdown = [linear_time, attn_time, total_communication_delay]

    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=prefill_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                )
