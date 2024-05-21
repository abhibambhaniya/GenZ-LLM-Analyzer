Model_list = ['LLaMA_7b', 'mixtral_7x8', 'LLaMA_70b', 'gpt-3', 'gpt-4']
Model_names_in_plot = ['LLaMA2-07B', 'Mixtral-8x7B', 'LLaMA3-70B', 'GPT3-175B', 'GPT4-1.8T'] ## Only to be used for plot

## Format: Batch, Beam, Input Tokens, Output Tokens, Latency(s)
Prefill_requirements = {
        'QA':(1,4,1000,200,0.2),
        'Chat Bots':(1,2,3000,1000,0.2), 
        'QA + RAG':(1,4,10000,200,0.4),
        'Summarization':(1,4,15000,1000,2), 
        'Code Generation':(1,4,20000,50,5)
        }

Decode_requirements = {
        'QA':(1,4,1000,200,0.010), 
        'Chat Bots':(1,2,3000,1000,0.010), 
        'QA + RAG':(1,4,10000,200,0.010), 
        'Summarization':(1,4,15000,1000,0.020), 
        'Code Generation':(1,4,20000,50,0.020)
        }

system_list = [
   'TPUv5e', 'H100', 'MI300X', 'gaudi3',   ## Real Systems
{'Flops':300, 'Memory_BW':600, 'Memory_size':16, 'ICN':400, 'real_values':True},    # QA
{'Flops':900, 'Memory_BW':3600, 'Memory_size':16, 'ICN':400, 'real_values':True},    # Chat Bots
{'Flops':1800, 'Memory_BW':2500, 'Memory_size':24, 'ICN':400, 'real_values':True},    # QA + RAG
{'Flops':750, 'Memory_BW':1875, 'Memory_size':48, 'ICN':400, 'real_values':True},    # Summarization
{'Flops':1000, 'Memory_BW':1250, 'Memory_size':24, 'ICN':400, 'real_values':True},    # Code Generation
]

System_names_in_plot = ['TPUv5e', 'H100', 'MI300X', 'Gaudi3',
                         'Sys_QA','Sys_Chat', 'Sys_RAG_QA', 'Sys_Summ', 'Sys_Code']

import plotnine as p9
plot_theme = p9.theme(axis_text_x=p9.element_text(size=24),
              axis_text_y=p9.element_text(size=24),
              axis_title_x=p9.element_text(size=24),  # Adjust the size as needed
              axis_title_y=p9.element_text(size=24),
              plot_title=p9.element_text(size=24),  # Adjust the size as needed
              # legend_position='inside', 
              legend_position='right',
              # legend_key_spacing_y=20000,
              legend_title=p9.element_text(size=24,margin={'b': 10}),        
              legend_text=p9.element_text(size=24),  # Adjust the size as needed
              panel_background=p9.element_rect(fill='white'),
              plot_background=p9.element_rect(fill='white'),
              panel_border=p9.element_rect(color='black', size=1),
              panel_grid_major=p9.element_line(color='black', size=0.5),
              panel_grid_minor=p9.element_line(color='black', size=0.25, linetype='dashed'),
              axis_line=p9.element_line(color='white', size=1),
              strip_text=p9.element_text(size=24),
              figure_size=(20, 6))

