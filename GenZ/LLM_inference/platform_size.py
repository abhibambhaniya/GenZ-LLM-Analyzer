import numpy as np
from GenZ.LLM_inference import decode_moddeling, prefill_moddeling, get_best_parallization_strategy, get_inference_system 

def get_minimum_system_size(
        stage='decode', model='llama_7b', max_batch_size = 1, beam_size = 1,
        input_tokens = 2000, output_tokens = 256,
        system_name = {'Flops': 200, 'Memory_size': 32, 'Memory_BW': 1000, 'ICN': 300 , 'real_values':True},
        bits='bf16', debug=False
        ):

    if stage=='prefill':
        model_df, summary_table = prefill_moddeling(model = model, batch_size = max_batch_size,
                            input_tokens = input_tokens, output_tokens = output_tokens, model_profilling=True,
                            tensor_parallel = 1, pipeline_parallel = 1, bits=bits, debug=debug)
    elif stage=='decode':
        model_df, summary_table = decode_moddeling(model = model, batch_size = max_batch_size, Bb = beam_size ,
                            input_tokens = input_tokens, output_tokens = output_tokens, model_profilling=True,
                            tensor_parallel = 1, pipeline_parallel = 1, bits=bits, debug=debug)
    Total_memory_required = (summary_table.loc[0,'Model Weights (MB)'] + summary_table.loc[0,'KV Cache (MB)'])   ## MBs
    system = get_inference_system(system_name =system_name , bits = bits)
    Node_memory_size = system.get_off_chip_mem_size()       ## In MBs
    Num_nodes = Total_memory_required / Node_memory_size
    if Num_nodes <= 1:
        Num_nodes = 1
    else:
        Num_nodes = int(np.power(2,np.ceil(np.log2(Num_nodes))))
    ##### RE verifying the memory sizes
    while 1:
        try:    ## Check if model fits
            _ = get_best_parallization_strategy(stage=stage, model=model, total_nodes=Num_nodes, batch_size = max_batch_size, beam_size = beam_size,
                        input_tokens = input_tokens, output_tokens = output_tokens,
                        system_name = system_name, bits=bits, debug=debug).sort_values(by='Tokens/s', ascending=False)
            break
        except: ## If model doesn't fit, have 2x number of cores.
            Num_nodes *= 2

    return Num_nodes