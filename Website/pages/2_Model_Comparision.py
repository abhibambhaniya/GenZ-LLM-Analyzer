# Import necessary libraries
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from GenZ import decode_moddeling, prefill_moddeling, get_configs
import pandas as pd
from tqdm import tqdm
import time

from Systems.system_configs import system_configs

st.set_page_config(
    page_title="Model Comparisons",
    page_icon="🔬",

    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues',
        'Report a bug': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues",
        'About': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/blob/main/README.md"
    }
)

st.sidebar.title("Model comparisons")
st.sidebar.subheader("1. Select Models to compare")
st.sidebar.subheader("2. Select Preconfigured use-case or make your customized use-case")
st.sidebar.subheader("3. Select HW System to run on")

st.sidebar.info(
    "This app is maintained by Abhimanyu Bambhaniya. ")

st.sidebar.info("If this app helps you, consider giving it a star! [⭐️](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer)")
st.title("Compare various Model Performance")

# Define the function to generate the demand curve
def generate_demand_curve(system_box, system_eff, num_nodes_slider,
                        model_box, quantization_box, batch_slider,
                        input_token_slider, output_token_slider, beam_size,
                        ):
    warnings.filterwarnings("ignore")

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

    data_df = pd.DataFrame(data, columns = ['Model', 'Stage','Batch', 'Latency(ms)', 'Tokens/s'])
    chip_req_df = pd.DataFrame(mem_size_data, columns = ['Model', 'NPU memory','Batch', 'Beam size', 'Input Tokens', 'Output Tokens', 'Min. Chips'])

    data_df['Stage'] = pd.Categorical(data_df['Stage'], categories=['Prefill','Decode'])

    fig = px.line(data_df, x="Batch", y="Tokens/s",  line_group="Model", color="Model",
                facet_row='Stage', facet_row_spacing = 0.1,
                labels={"Batch": "Batch", "Tokens/s": "Tokens/s", "Model": "Model"},
                width=1200, height=600, markers=True)


    # Customize axis labels
    fig.update_xaxes(title_font=dict(size=24))
    fig.update_yaxes(title_font=dict(size=24))

    # Customize tick labels
    fig.update_xaxes(tickfont=dict(size=24), linecolor='white',)
    fig.update_yaxes(tickfont=dict(size=24))
    fig.update_yaxes(matches=None, linecolor='white',)

    # # Customize facet labels
    fig.update_layout(
        font_size=24,
        # plot_bgcolor='rgb(127,127,127)',
        legend = dict(font = dict(family = "Courier", size = 24),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
        legend_title = dict(font = dict(family = "Courier", size = 24))


        )
    fig.update_traces(marker=dict(size=12,
                    line=dict(width=4,)),
                # selector=dict(mode='markers')
                )

    if len(mem_size_data) == 0:
        return fig
    elif len(data) == 0:
        return chip_req_df
    else:
        return fig, chip_req_df

def main():

    regenerate_plot = True

    col1, col2, col3 = st.columns([6,3,4])
    tabs_font_css = """
    <style>
    div[class*="stTextArea"] label p {
    font-size: 20px;
    }

    div[class*="stTextInput"] label p {
    font-size: 20px;
    }

    div[class*="stNumberInput"] label p {
    font-size: 20px;
    }

    div[class*="stSlider"] label p {
    font-size: 20px;
    }

    div[class*="stSelectbox"] label p {
    font-size: 20px;
    }

    div[class*="stMarkdown"] label p {
    font-size: 20px;
    }

    div[class*="stMultiSelect"] label p {
    font-size: 20px;
    }

    .big-font {
    font-size:24px !important;
    }
    </style>
    """

    st.write(tabs_font_css, unsafe_allow_html=True)
    with col1:
        st.header("Model")
        if 'models' not in st.session_state:
            st.switch_page("Home.py")
        selected_models = st.multiselect("Models:", st.session_state.models, default=st.session_state.models[0])
        st.markdown("""
            <style>
                .stMultiSelect [data-baseweb=select] span{
                    max-width: 250px;
                    font-size: 1rem;
                }
            </style>
            """, unsafe_allow_html=True)
        quantization = st.selectbox("Quantization:", ['fp8', 'bf16', 'int8', 'int4', 'int2', 'f32'])

    with col2:
        st.header("Use case")

        max_batch_size = st.number_input("Max Batch Size:", value=8, step=1,min_value=1)
        use_case = st.selectbox("Usecases:", ['Ques-Ans', 'Text Summarization', 'Chatbots', 'Code Gen.', 'Custom'])
        if 'Ques-Ans' == use_case:
            used_beam_size = 4
            used_input_tokens = 1000
            used_output_tokens = 200
        elif 'Text Summarization' == use_case:
            used_beam_size = 4
            used_input_tokens = 15000
            used_output_tokens = 1000
        elif 'Chatbots' == use_case:
            used_beam_size = 2
            used_input_tokens = 2048
            used_output_tokens = 128
        elif 'Code Gen.' == use_case:
            used_beam_size = 4
            used_input_tokens = 20000
            used_output_tokens = 50
        else:
            used_beam_size = 4
            used_input_tokens = 1000
            used_output_tokens = 200
        beam_size = st.slider("No. of Parallel Beams:", min_value=1, max_value=16, value=used_beam_size)
        input_tokens = st.number_input("Input Tokens:", value=used_input_tokens, step=100)
        output_tokens = st.number_input("Output Tokens:", value=used_output_tokens, step=100)

    with col3:
        st.header("HW System")
        if 'systems' not in st.session_state:
            st.switch_page("Home.py")
        selected_system = st.selectbox("System:", st.session_state.systems)
        nodes = st.number_input("# Nodes:", value=2, step=1)
        system_efficiency = st.slider("System Efficiency:", min_value=0.0, max_value=1.0, value=0.80, step=0.01)

        if selected_system in system_configs:
            current_system_config = system_configs[selected_system]
            used_flops = current_system_config.get('Flops', '')
            used_mem_bw = current_system_config.get('Memory_BW', '')
            used_mem_cap = current_system_config.get('Memory_size', '')
            used_icn_bw = current_system_config.get('ICN', '')
        else:
            used_flops = 312
            used_mem_bw = 1600
            used_mem_cap = 40
            used_icn_bw = 150
        flops = st.number_input("FLOPS (TOPS):", value=used_flops, step=100)
        mem_bw = st.number_input("MEM BW (GB/s):", value=used_mem_bw, step=100)
        mem_cap = st.number_input("Mem Capacity (GBs):", value=used_mem_cap, step=8)
        icn_bw = st.number_input("ICN BW (GB/s):", value=used_icn_bw, step=10)

    # Create Plotly bar chart
    if selected_models:
        outputs = generate_demand_curve(
            system_box = {'Flops': flops, 'Memory_BW': mem_bw, 'Memory_size': mem_cap, 'ICN': icn_bw , 'real_values':True},
            system_eff = system_efficiency,
            num_nodes_slider = nodes,
            model_box=selected_models,
            quantization_box=quantization,
            batch_slider=max_batch_size,
            input_token_slider=input_tokens,
            output_token_slider=output_tokens,
            beam_size = beam_size
            )
        with st.status("Computing metric...", expanded=True):
            st.write("Building Platforms...")
            time.sleep(1)
            st.write("Getting LLM inference analysis...")
            time.sleep(0.5)
            st.write("Generating charts...")
            time.sleep(0.5)
        if isinstance(outputs, pd.DataFrame):
            st.write("Number of nodes is insufficient, please increase the nodes to fit the model")
            st.dataframe(outputs)
        elif isinstance(outputs, go.Figure):
            st.plotly_chart(outputs)
        else:
            st.plotly_chart(outputs[0])
            st.write("Number of nodes is insufficient, please increase the nodes to fit the model")
            st.dataframe(outputs[1])

        regenerate_plot = False


    # Display some calculated metrics
    # st.subheader("Calculated Metrics")
    # st.write(f"Effective System Performance: {flops}, {mem_bw}")

if __name__ == "__main__":
    main()
