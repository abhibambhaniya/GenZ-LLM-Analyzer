from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import get_configs, create_full_prefill_model

unit = Unit()

def prefill_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    output_tokens = 0,          ## Only for prefill
    system_name = 'A100_40GB_GPU', system_eff=1, bits='bf16', debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1, return_model_df=False,
    model_offload = False):

    ##################################################################################################
    ### Model parsing
    ##################################################################################################

    model_config = get_configs(model)

    model_D = model_config.hidden_size
    F = model_config.intermediate_size
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    num_layers = model_config.num_decoder_layers
    is_moe = model_config.moe_layer_freq

    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    system = get_inference_system(system_name = system_name, bits = bits, ceff=system_eff, meff=system_eff)

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_prefill = create_full_prefill_model(input_sequence_length=input_tokens, name=model,
                                                    tensor_parallel=tensor_parallel)


    model_df = get_model_df(model_prefill, system, unit, batch_size, intermediate_on_chip=True , model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)
    # summary_table_cols = [f'MACs ({unit.unit_flop})', f'Total Data ({unit.unit_mem})']
    # ## Drop columns not is list
    # summary_table = summary_table[summary_table.columns.intersection(summary_table_cols)]

    # model_weights = 0
    # kv_cache = 0
    # for i in range(len(model_df)):
    #     if ('Logit'  in model_df.loc[i, 'Op Type']  or 'Attend'  in model_df.loc[i, 'Op Type']):
    #         kv_cache += model_df.loc[i,'Input_w (MB)']
    #     else:
    #         model_weights = model_weights + model_df.loc[i,'Input_w (MB)']

    # num_layers_per_pipeline_stage = num_layers // pipeline_parallel
    ## TP data volumn
    # single_layer_all_reduce_data  = 2* batch_size*input_tokens*model_D* system.get_bit_multiplier(type='M')
    # total_all_reduce_data = single_layer_all_reduce_data * num_layers_per_pipeline_stage
    ## PP volumn
    single_stage_pipe_data =  batch_size*input_tokens*model_D* system.get_bit_multiplier(type='M')
    total_pipe_data = single_stage_pipe_data * (pipeline_parallel-1)




    model_weights = summary_table[f'Total Weights ({unit.unit_mem})'].values[0]        ## In MB
    kv_cache = summary_table[f'KV Cache ({unit.unit_mem})'].values[0]                  ## In MB

    total_memory_req = model_weights + kv_cache
    num_nodes = pipeline_parallel * tensor_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if  per_chip_memory < total_memory_req:
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
    if num_layers % pipeline_parallel != 0:
        raise ValueError(f"Number of layers:{num_layers} should be divisible by PP:{pipeline_parallel}")

    if model_profilling:
        return model_df, summary_table

    ##################################################################################################
    ### Prefill generation time
    ##################################################################################################
    model_prefill = create_full_prefill_model(input_sequence_length=input_tokens, name=model,
                                        tensor_parallel=tensor_parallel)
    model_df = get_model_df(model_prefill, system, unit, batch_size, intermediate_on_chip=True )
    summary_table = get_summary_table(model_df, unit)
    prefill_stage_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]                 # Latency in millisec
    if return_model_df:
        return model_df, summary_table
    if debug:
        display_df(model_df)
        display(summary_table)

    ##################################################################################################
    ### Communication time
    ##################################################################################################
    # ## TP time
    # if tensor_parallel > 1:
    #     all_reduce_delay  =  2*get_AR_time(data = single_layer_all_reduce_data/2 ,num_AR_nodes=tensor_parallel, system=system)
    # else:
    #     all_reduce_delay = 0

    ## PP time
    single_stage_pipe_delay = get_message_pass_time(data = single_stage_pipe_data, system=system )

    ## Total Comm time
    total_communication_delay = single_stage_pipe_delay * (pipeline_parallel-1)


    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    prefill_latency =  prefill_stage_latency * pipeline_parallel + single_stage_pipe_delay * (pipeline_parallel-1)

    if debug:
            print(f'Prefill Latency:{prefill_latency} {unit.unit_time}')
            print(f'Single Pipe Stage:{prefill_stage_latency}  {unit.unit_time}')
            # print(f'Layers per pipeline stage:{(num_layers_per_pipeline_stage)}')


    thrpt = 1000 * batch_size / prefill_latency        ## this is for TP
    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    if pipeline_parallel > 1:
        token_generation_interval = prefill_stage_latency + single_stage_pipe_delay
        thrpt = 1000 * batch_size / token_generation_interval

    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]
    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]
    total_communication_delay += summary_table[f'Comm Latency ({unit.unit_time})'].values[0]
    runtime_breakdown = [linear_time, attn_time, total_communication_delay]

    ##################################################################################################
    ### Output Generation
    ##################################################################################################
    # Error_rate = 100*(total_time-prefill_latency)/prefill_latency
    # if Error_rate > 5:
        # raise ValueError(f"Error in latency calc. Prefill Latency:{prefill_latency} msec , Latency based on last token : {total_time} msec, \n Attn time:{attn_time}; Linear time:{linear_time}; AR time:{all_reduce_delay * (num_layers//pipeline_parallel)}; Pipeline Comm time:{single_stage_pipe_delay * (pipeline_parallel-1)}") 

    # if debug:
    #     print(f'Error = {Error_rate} in latency calc. Prefill Latency:{prefill_latency} msec , Latency based on last token : {total_time} msec')
    #     print(f'Attn time:{attn_time}; Linear time:{linear_time}; AR time:{all_reduce_delay * (num_layers_per_pipeline_stage)}; Pipeline Comm time:{single_stage_pipe_delay * (pipeline_parallel-1)}')

    return ModdelingOutput(
                        Latency=prefill_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                )
