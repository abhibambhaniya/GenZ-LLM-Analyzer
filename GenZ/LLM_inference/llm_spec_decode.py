from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_chunked_model, create_full_prefill_model, create_full_decode_model
from math import ceil

unit = Unit()


def spec_prefill_modeling(model = 'meta-llama/Llama-3.1-70B', draft_model = 'meta-llama/meta-llama-3.1-8b',
    batch_size = 1, 
    input_tokens = 1024,       # Input context tokens
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
    model_full_created = create_full_prefill_model(name=model,
                                            input_sequence_length= input_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_model_df = get_model_df(model_full_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    full_summary_table = get_summary_table(full_model_df, unit, model_characterstics = True)

    full_model_weights = full_summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    full_model_kv_cache = full_summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB
    
    # While draft model can be fit with a different parallelism, right now we are assuming the same parallelism for both draft and full model.
    model_draft_created = create_full_prefill_model(name=draft_model,
                                            input_sequence_length= input_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_model_df = get_model_df(model_draft_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    draft_summary_table = get_summary_table(draft_model_df, unit, model_characterstics = True)

    draft_model_weights = draft_summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    draft_kv_cache = draft_summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB    
    
    total_memory_req = full_model_weights + full_model_kv_cache + draft_model_weights + draft_kv_cache

    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if  per_chip_memory * pipeline_parallel < total_memory_req:
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
        return pd.concat([full_model_df, draft_model_df]), pd.concat([full_summary_table, draft_summary_table])

    ##################################################################################################
    ### Initial prefill times
    ##################################################################################################
    model_prefill = create_full_prefill_model(  name=model,
                                            input_sequence_length=input_tokens,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_prefill, system, unit, ub,  intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)

    full_model_prefill_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if debug:
        print("Full Model Prefill")
        display_df(simplify_df(model_df))
        display(summary_table)
    
    model_draft_prefill = create_full_prefill_model(  name=draft_model,
                                            input_sequence_length=input_tokens,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_draft_prefill, system, unit, ub,  intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)

    draft_model_prefill_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if debug:
        print("Draft Model Prefill")
        display_df(simplify_df(model_df))
        display(summary_table) 
    
    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    prefill_latency = full_model_prefill_latency + draft_model_prefill_latency
    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    
    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    if pipeline_parallel > 1:
        micro_batch_latency = prefill_latency
        ## If the N micro batches, then the total latency is (N-1)*stage latency + initial_latency
        ## We make the assumption that the pipeline is balanced and the latency is same for all stages
        total_latency = ((num_micro_batches-1) * (prefill_latency / pipeline_parallel)) + micro_batch_latency
        thrpt = 1000 * batch_size / total_latency
    else:
        thrpt = 1000 * batch_size / prefill_latency

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
                        Latency=prefill_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                )
    
def spec_decode_modeling(model = 'meta-llama/Llama-3.1-70B', draft_model = 'meta-llama/meta-llama-3.1-8b',
    batch_size = 1, 
    input_tokens = 1024,       # Input context tokens
    output_tokens = 1024,      # Output context to be generated
    token_acceptance_rate = 0.7,            # Probability of accepting a token from draft model decoding.
    # This gamma parameter in the paper: arxiv.org/pdf/2211.17192
    num_parallel_tokens = 8,    # Number of tokens to be decoded in parallel by the full model.
                                # This means after num_parallel_tokens decode steps of the draft model, num_parallel_tokens tokens are checked in parallel by the full model.
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

    assert num_parallel_tokens > 1, "Number of parallel tokens must be > 1, for the full model to be useful"
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
    model_full_created = create_full_decode_model(name=model,
                                            input_sequence_length= input_tokens,
                                            output_gen_tokens = output_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_model_df = get_model_df(model_full_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    full_summary_table = get_summary_table(full_model_df, unit, model_characterstics = True)

    full_model_weights = full_summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    full_model_kv_cache = full_summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB
    
    # While draft model can be fit with a different parallelism, right now we are assuming the same parallelism for both draft and full model.
    model_draft_created = create_full_decode_model(name=draft_model,
                                            input_sequence_length= input_tokens,
                                            output_gen_tokens= output_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_model_df = get_model_df(model_draft_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    draft_summary_table = get_summary_table(draft_model_df, unit, model_characterstics = True)

    draft_model_weights = draft_summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    draft_kv_cache = draft_summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB    
    
    total_memory_req = full_model_weights + full_model_kv_cache + draft_model_weights + draft_kv_cache

    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if  per_chip_memory * pipeline_parallel < total_memory_req:
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
        return pd.concat([full_model_df, draft_model_df]), pd.concat([full_summary_table, draft_summary_table])

    ##################################################################################################
    ### Model decode times
    ##################################################################################################
    model_draft_decode = create_full_decode_model(  name=draft_model,
                                            input_sequence_length=input_tokens,
                                            output_gen_tokens = output_tokens ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_draft_decode, system, unit, ub,  intermediate_on_chip=True )
    draft_summary_table = get_summary_table(model_df, unit)

    draft_model_decode_latency = draft_summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if debug:
        print("Draft Model Decode")
        display_df(simplify_df(model_df))
        display(draft_summary_table) 

    # For full model, the exsisting KV cache is input tokens + output tokens, and
    # we are checking num_parallel_tokens tokens in parallel.
    model_decode = create_full_chunked_model(  name=model,
                                            prefill_kv_sizes = [(input_tokens+output_tokens, num_parallel_tokens)],
                                            decode_kv_sizes = [] ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    model_df = get_model_df(model_decode, system, unit, ub,  intermediate_on_chip=True )
    full_summary_table = get_summary_table(model_df, unit)

    full_model_decode_latency = full_summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if debug:
        print("Full Model Decode")
        display_df(simplify_df(model_df))
        display(full_summary_table)
    
        
    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    # num_parallel_tokens = 4
    # token_acceptance_rate = 0.7
    # 4 draft token generated.
    # Chance of 1 token accepted = 0.7     , Chance of reject = 1-0.7
    # Chances of 2 token accepted = 0.7**2 , Chance of 2 token rejected = 1-0.7**2
    # Chances of 3 token accepted = 0.7**3 , Chance of 3 token rejected = 0.3*0.7**2
    # Chance of 4 token accepeted= 0.7**4 , Chance of 4 token rejected = 0.3*0.7**3
    # Latency = (4*full_model_decode_latency) + draft_model_decode_latency
    # Number of tokens generated = 4*(x**4) + 3*(x**3) + 2*(x**2) + 1*(x**1) = 4*(1-x)**3

    total_latency =  full_model_decode_latency + num_parallel_tokens * draft_model_decode_latency
    # 
    tokens_generated = expected_tokens(num_parallel_tokens, token_acceptance_rate)
    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    if pipeline_parallel > 1:
        micro_batch_latency = total_latency
        ## If the N micro batches, then the total latency is (N-1)*stage latency + initial_latency
        ## We make the assumption that the pipeline is balanced and the latency is same for all stages
        total_latency = ((num_micro_batches-1) * (total_latency / pipeline_parallel)) + micro_batch_latency
        thrpt = 1000 * batch_size * tokens_generated / total_latency
    else:
        thrpt = 1000 * batch_size * tokens_generated / total_latency

    summary_table = draft_summary_table + full_summary_table
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
                        Latency=total_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                        tokens_generated=tokens_generated
                )
    
def expected_tokens(N: int, x: float) -> float:
    """
    Calculate expected number of accepted tokens in speculative decoding.
    
    Args:
        N (int): Number of tokens generated speculatively
        x (float): Probability of token acceptance (between 0 and 1)
    
    Returns:
        float: Expected number of accepted tokens
    """
    if not 0 <= x <= 1:
        raise ValueError("Probability x must be between 0 and 1")
    if N < 1:
        raise ValueError("N must be positive")
    
    # For k < N: k tokens accepted with prob x^k * (1-x)
    # For k = N: N tokens accepted with prob x^N
    
    # We start from 1 because, the full model will correct 1 token if it is wrong.
    expected = 1
    for k in range(1, N):
        expected += k * (x**k) * (1-x)
    expected += N * (x**N)
    
    return min(expected, N)