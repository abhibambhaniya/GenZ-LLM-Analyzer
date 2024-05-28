from .utils import ModdelingOutput, get_inference_system, get_offload_system
import os, sys
script_dir = os.getcwd()
module_path = script_dir
for _ in range(5):
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
    if os.path.basename(module_path) =='roofline':
        break
from src.unit import Unit
from src.operators import *

from src.operator_base import op_type_dicts
from src.system import System
import pandas as pd
from src.analye_model import *
import warnings
from src.collective_times import *
from utils.plot_rooflines import *

data_path = os.path.join(module_path,"data")
model_path = os.path.join(data_path,"model")
unit = Unit()

def prefill_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    output_tokens = 0, FLAT = True,         ## Only for prefill
    system_name = 'A100_40GB_GPU', system_eff=1, bits='bf16', debug= False, model_profilling = False, 
    tensor_parallel = 1, pipeline_parallel = 1, time_breakdown = False, return_model_df=False,
    model_offload = False):
    
    ################################################################################################## # 
    ### Model parsing
    ################################################################################################## # 

    model_config = get_configs(model, get_model_config=True)
    
    model_D = model_config.hidden_size
    F = model_config.intermediate_size 
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    num_layers = model_config.num_decoder_layers
    is_moe = model_config.moe_layer_freq

    ################################################################################################## # 
    ### System Declaration
    ################################################################################################## # 

    system = get_inference_system(system_name = system_name, bits = bits, ceff=system_eff, meff=system_eff)
   
    ################################################################################################## # 
    ### Model Characterization Calculation
    ################################################################################################## # 
    model_prefill = create_inference_moe_prefix_model(input_sequence_length=input_tokens,output_gen_tokens = 0 , 
                                        name=model,data_path=data_path, Hkv=Hkv, tensor_parallel=tensor_parallel)


    model_df = get_model_df(model_prefill, system, unit, batch_size, data_path, intermediate_on_chip=FLAT , model_characterstics = True)
    summary_table = get_summary_table(model_df,system,unit, model_characterstics = True)
    summary_table_cols = [f'MACs ({unit.unit_flop})', f'Total Data ({unit.unit_mem})']
    ## Drop columns not is list
    summary_table = summary_table[summary_table.columns.intersection(summary_table_cols)]

    model_weights = 0
    kv_cache = 0
    for i in range(len(model_df)):
        if ('Logit'  in model_df.loc[i, 'Op Type']  or 'Attend'  in model_df.loc[i, 'Op Type']):
            kv_cache += model_df.loc[i,'Input_w (MB)'] 
        else:
           model_weights = model_weights + model_df.loc[i,'Input_w (MB)'] 

    num_layers_per_pipeline_stage = num_layers // pipeline_parallel
    ## TP data volumn
    single_layer_all_reduce_data  = 2* batch_size*input_tokens*model_D* system.get_bit_multiplier(type='M')
    total_all_reduce_data = single_layer_all_reduce_data * num_layers_per_pipeline_stage
    ## PP volumn
    single_stage_pipe_data =  batch_size*input_tokens*model_D* system.get_bit_multiplier(type='M')
    total_pipe_data = single_stage_pipe_data * (pipeline_parallel-1)

    model_weights *= num_layers_per_pipeline_stage
    kv_cache *= num_layers_per_pipeline_stage

    summary_table[f'MACs ({unit.unit_flop})'] = summary_table[f'MACs ({unit.unit_flop})'].apply(lambda x: x*num_layers_per_pipeline_stage)
    summary_table[f'Total Data ({unit.unit_mem})'] = summary_table[f'Total Data ({unit.unit_mem})'].apply(lambda x: x*num_layers_per_pipeline_stage)
    summary_table[f'Model Weights ({unit.unit_mem})'] = model_weights       ## In MB
    summary_table[f'Unused Weights ({unit.unit_mem})'] = 0       ## In MB
    summary_table[f'KV Cache ({unit.unit_mem})'] = kv_cache                 ## In MB
    summary_table[f'AR data ({unit.unit_mem})'] = unit.raw_to_unit( total_all_reduce_data, 'M')      ## In MB
    summary_table[f'Pipe data  ({unit.unit_mem})'] = unit.raw_to_unit( total_pipe_data, 'M')         ## In MB

    total_memory_req = model_weights + kv_cache
    Num_cores = pipeline_parallel * tensor_parallel

    ################################################################################# # 
    ### Offloading calculations
    ################################################################################# # 
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

    ################################################################################################## # 
    ### Prefill generation time
    ################################################################################################## # 
    model_prefill = create_inference_moe_prefix_model(input_sequence_length=input_tokens,output_gen_tokens = 0 , 
                                        name=model,data_path=data_path, Hkv=Hkv, tensor_parallel=tensor_parallel)
    model_df = get_model_df(model_prefill, system, unit, batch_size, data_path, intermediate_on_chip=FLAT )

    # if debug:
        # display_df(model_df)
    summary_table = get_summary_table(model_df,system,unit)
    prefill_latency = summary_table['Latency (msec)'].values[0]   # Latency in millisec
    if return_model_df:
        return model_df, summary_table
    if debug:
        display_df(model_df)
        display(summary_table)

    ################################################################################################## # 
    ### Communication time
    ################################################################################################## # 
    ## TP time
    if tensor_parallel > 1:
        all_reduce_delay  =  2*get_AR_time(data = single_layer_all_reduce_data/2 ,num_AR_nodes=tensor_parallel, system=system) 
    else:
        all_reduce_delay = 0

    ## PP time
    single_stage_pipe_delay = get_message_pass_time(data = single_stage_pipe_data, system=system )

    ## Total Comm time
    total_communication_delay = single_stage_pipe_delay * (pipeline_parallel-1) + all_reduce_delay * num_layers


    ################################################################################################## # 
    ### Final Latency and Thrpt Calculation
    ################################################################################################## # 
    
    ## Single layer will have compute/memory time + 2 AR delay
    single_layer_time = prefill_latency + all_reduce_delay 
    single_pipe_stage = single_layer_time * num_layers_per_pipeline_stage

    prefill_latency =  single_pipe_stage * pipeline_parallel + single_stage_pipe_delay * (pipeline_parallel-1)

    if debug:
            print(f'Prefill Latency:{prefill_latency} {unit.unit_time}')
            print(f'single_pipe_stage:{single_pipe_stage}  {unit.unit_time}; single_layer_time:{single_layer_time}  {unit.unit_time}')
            print(f'Layers per pipeline stage:{(num_layers_per_pipeline_stage)}')
        

    thrpt = 1000 * batch_size / prefill_latency        ## this is for TP
    ## 1000x because the latency is in milli seconds. thrpt is in Token/s
    if pipeline_parallel > 1:
        token_generation_interval = single_pipe_stage + single_stage_pipe_delay 
        thrpt = 1000 * batch_size / token_generation_interval 

    attn_time, linear_time = 0,0
    for i in range(len(model_df)):
        if ('Logit' in model_df.loc[i, 'Op Type'] or 'Attend' in model_df.loc[i, 'Op Type']):
            attn_time +=  model_df.loc[i,'Latency (msec)']
            # print(i, model_df.loc[i, 'Op Type'])
        else:
            linear_time +=  model_df.loc[i,'Latency (msec)']
            
    linear_time *= num_layers     ## In milliseconds
    attn_time *= num_layers       ## In milliseconds
    total_time = linear_time + attn_time + total_communication_delay
    runtime_breakdown = [linear_time, attn_time, total_communication_delay]

    ################################################################################################## # 
    ### Output Generation
    ################################################################################################## # 
    Error_rate = 100*(total_time-prefill_latency)/prefill_latency
    # if Error_rate > 5:
        # raise ValueError(f"Error in latency calc. Prefill Latency:{prefill_latency} msec , Latency based on last token : {total_time} msec, \n Attn time:{attn_time}; Linear time:{linear_time}; AR time:{all_reduce_delay * (num_layers//pipeline_parallel)}; Pipeline Comm time:{single_stage_pipe_delay * (pipeline_parallel-1)}") 
    
    if debug:
        print(f'Error = {Error_rate} in latency calc. Prefill Latency:{prefill_latency} msec , Latency based on last token : {total_time} msec')
        print(f'Attn time:{attn_time}; Linear time:{linear_time}; AR time:{all_reduce_delay * (num_layers_per_pipeline_stage)}; Pipeline Comm time:{single_stage_pipe_delay * (pipeline_parallel-1)}')

    return ModdelingOutput(
                        Latency=prefill_latency,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                )
    