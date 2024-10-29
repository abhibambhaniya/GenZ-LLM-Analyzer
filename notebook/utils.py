# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from IPython.display import clear_output, display

import warnings

from GenZ import decode_moddeling, prefill_moddeling, get_configs
import pandas as pd
from tqdm import tqdm

from Systems.system_configs import system_configs


# Define the function to generate the demand curve
def generate_demand_curve(graph_type, system_box, system_eff, num_nodes_slider,
                        model_box, quantization_box, batch_slider,
                        input_token_slider, output_token_slider, beam_size,
                        flops, mem_bw, mem_cap,icn_bw):
    warnings.filterwarnings("ignore")
    clear_output()
    if system_box == 'Custom':
        system_box = {'Flops': flops, 'Memory_size': mem_cap, 'Memory_BW': mem_bw, 'ICN': icn_bw , 'real_values':True}  
    else:
        if isinstance(system_box, str) and system_box in system_configs:
            system_box = system_configs[system_box]
        elif not isinstance(system_box, dict):
            raise ValueError(f'System mentioned:{system_box} not present in predefined systems. Please use systems from Systems/system_configs')

    # ('1. ISO-HW: Model vs Throughput', 1),
    # ('2. ISO-HW: Model vs Latency (TTFT, TPOT)', 2),
    # ('3. ISO-HW, ISO-Model: Batch vs Throughput/Latency', 3)
    # ('4. ISO-Usecase, Multiple HW, Throuput vs Latency', 4)

    data = []
    mem_size_data = []
    batch_size_list = [1,2,4,8,16,32,48,64,80,96,112,128,136,144,160, 172, 180, 200, 224, 240, 256]
    for batch_size in tqdm(batch_size_list):
        for model in model_box:
            if batch_size <= batch_slider:
                model_name = get_configs(model).model
                try:
                    prefill_outputs = prefill_moddeling(model = model, batch_size = batch_size,
                                            input_tokens = input_token_slider,
                                            system_name = system_box, system_eff = system_eff,
                                            bits=quantization_box,
                                            tensor_parallel = num_nodes_slider, debug=False)
                    data.append([model_name,'Prefill',batch_size, prefill_outputs['Latency'], prefill_outputs['Throughput']])
                    decode_outputs = decode_moddeling(model = model, batch_size = batch_size, Bb = beam_size ,
                                            input_tokens = input_token_slider, output_tokens = output_token_slider,
                                            system_name = system_box, system_eff=system_eff,
                                            bits=quantization_box,
                                            tensor_parallel = num_nodes_slider, debug=False)
                    data.append([model_name,'Decode',batch_size,  decode_outputs['Latency'], decode_outputs['Throughput']])
                except:
                    # ValueError
                    decode_outputs, decode_summary_table = decode_moddeling(model = model, batch_size = batch_size, Bb = beam_size ,
                                            input_tokens = input_token_slider, output_tokens = output_token_slider,
                                            system_name = system_box, system_eff = system_eff,
                                            bits=quantization_box, model_profilling=True)
                    total_memory = int(system_box.get('Memory_size'))*1024  ## per device memory
                    memory_req =  decode_summary_table['Total Weights (MB)'].values[0] + decode_summary_table['KV Cache (MB)'].values[0]

                    mem_size_data.append([model, total_memory, batch_size, beam_size, input_token_slider, output_token_slider, np.ceil(memory_req/total_memory)])
    # assert len(data) > 0, "No Model fits in the given # of GPUs. Increase GPUs or use different Model"

    data_df = pd.DataFrame(data, columns = ['Model', 'Stage','Batch', 'Latency(ms)', 'Tokens/s'])
    chip_req_df = pd.DataFrame(mem_size_data, columns = ['Model', 'NPU memory','Batch', 'Beam size', 'Input Tokens', 'Output Tokens', 'Min. Chips'])
    if len(data) == 0 :
        # display(chip_req_df)
        return chip_req_df
    else:
        data_df['Stage'] = pd.Categorical(data_df['Stage'], categories=['Prefill','Decode'])

        if graph_type == 1:
            fig = px.line(data_df, x="Batch", y="Tokens/s",  line_group="Model", color="Model", facet_row='Stage',
                    labels={"Batch": "Batch", "Tokens/s": "Tokens/s", "Model": "Model"},
                    width=1200, height=600, markers=True)
        elif graph_type == 3:
            fig = px.line(data_df, x="Batch", y="Tokens/s",  line_group="Model", color="Model", facet_row='Stage',
                    labels={"Batch": "Batch", "Tokens/s": "Tokens/s", "Model": "Model"},
                    width=1200, height=600, markers=True)

        # Customize axis labels
        fig.update_xaxes(title_font=dict(size=24))
        fig.update_yaxes(title_font=dict(size=24))

        # Customize tick labels
        fig.update_xaxes(tickfont=dict(size=24))
        fig.update_yaxes(tickfont=dict(size=24))

        fig.update_yaxes(matches=None)

        # # Customize facet labels
        fig.update_layout(
            # font_color="black",
            # title_font_color="black",
            # legend_title_font_color="black",
            font_size=24
        )

        return fig
        # fig.show()
        # display(fig)
        # display(data_df)

