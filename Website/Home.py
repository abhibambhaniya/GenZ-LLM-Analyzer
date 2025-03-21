import streamlit as st
import time
from PIL import Image
from GenZ.Models import MODEL_DICT
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


genz_overview = Image.open('./Website/GenZ Overview.jpg')
llm_parallization = Image.open('./Website/various parallelism.jpg')
platform_abstraction = Image.open('./Website/GenZ Platform abstraction.jpg')
st.session_state.models = MODEL_DICT.list_models()

st.session_state.systems = ['H100_GPU', 'A100_40GB_GPU', 'A100_80GB_GPU', 'GH200_GPU', 'TPUv4','TPUv5e', 'MI300X', 'Gaudi3', 'Custom']


st.set_page_config(
    page_title="GenZ-LLM Analyzer",
    page_icon="🔬",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues',
        'Report a bug': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues",
        'About': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/blob/main/README.md"
    }
)
st.sidebar.info("If this app helps you, consider giving it a star! [⭐️](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer)")
# # Add custom CSS to increase the font size of all text


st.write("# Welcome to GenZ-LLM Analyzer! 👋")

st.image(genz_overview, caption='GenZ Framework Overview', use_container_width=True)

st.write(
    """
    <style>
    div[data-testid="stMarkdownContainer"] p{
    font-size: 20px;
    }
    div[data-testid="stMarkdownContainer"] li{
    font-size: 20px;
    }

    div[data-testid="stMarkdownContainer"] ul{
    font-size: 20px;
    }

    div[data-testid="stImageCaption"]{
    font-size: 20px;
    }

    
    div[data-testid="stExpander"] details summary p{
    font-size: 24px;
    }
    </style>
    

    """,
    unsafe_allow_html=True
)
st.markdown("""GenZ is an analytical tool designed to estimate the platform performance when serving Large Language Model (LLM) workloads.
          
- GenZ can analyze the performance of LLM workloads on different platforms, which are defined as collections of accelerators (like GPUs or TPUs or custom asic) connected by an interconnect network.
- GenZ can simulate distributed LLM inference at the platform scale while exploring various model-level optimizations, a feature not supported by existing frameworks.

GenZ Components: GenZ has three main components:
1. Model Profiler: GenZ determines the exact shape of each operator in the LLM model for different use cases. It also models optimizations like kernel fusion and quantization.
2. NPU Characterizer: This component models the hardware capabilities of an individual accelerator, considering factors like compute FLOPS, memory bandwidth, and efficiency factors for both computation and memory access.
3. Platform Characterizer: This component models the entire inference platform, including multiple NPUs connected via an interconnect network. It considers network properties like link latency, bandwidth, and efficiency, along with parallelism strategies such as data parallel, tensor parallel, and pipeline parallel.
""")



with st.expander("Local installation"):
    st.markdown("""
#### Pip Installation
```sh
pip install genz-llm
```

#### Local Editable Installation
```sh
git clone https://github.com/abhibambhaniya/genz.git
cd genz
pip install -r requirements.txt 
pip install -e .
```
    """)

    # st.markdown("[Steps to run GenZ]")
# with st.expander("Run GenZ"):
#     

# with st.expander("Understanding GenZ Output"):
#     st.markdown("[Explanation of GenZ output and how to interpret it]")

# Models and Workloads
with st.expander("Models and Workloads"):
  st.markdown("""
GenZ supports a range of public models from various model providers like Meta, Google, Microsoft, X-ai, and OpenAI. Any LLM model present on huggingface can be ported to GenZ.

Here is the complete list of models that are currently supported on GenZ:
  """)
  for i, m in enumerate(st.session_state.models):
      st.markdown(f"{i}. [{m}](https://www.huggingface.co/{m})")


with st.expander("Use Case Optimizations"):

  st.markdown("### Quantization")
  st.markdown("""Quantization is a technique that helps run Large Language Models (LLMs) faster by reducing the precision of their weights and activations from 32-bit floating-point numbers to lower-precision data types such as 8-bit integers. This reduces the memory usage and computational requirements, allowing for faster inference times and smaller model sizes. By compressing the models, quantization enables LLMs to run on devices with limited memory and processing power, making them more suitable for edge computing, mobile applications, and real-time applications where speed is critical.
GenZ supports the following data types for running LLMs:

| Data Type | Bits |
|:---------:|:----:|
|    FP32   |  32  |
|    BF16   |  16  |
|  INT8/FP8 |   8  |
|  INT4/FP4 |   4  |
|    INT2   |   2  |
""")

  st.markdown("### Batching")
  st.markdown("""
Since LLMs in the decode phase are memory-bound, batching multiple queries can help improve the decode throughput.
Ex: Running a LLaMA2-7B model at int8 precision on NVIDIA's A100 (80GB) GPU, with 2k input tokens, as we increase the batch from 1 to 44, we see a 10.9x improvement in the decode throughput while first token latency only increases by 1.83x. 
    """)
  st.markdown("### Parallelization")
  st.image(llm_parallization, caption='Various parallelization strategies. Each grey box represents an accelerator NPUs and the colored blocks represent the layers. We illustrate how the LLM runs on different NPUs for each strategy. (a) In full TP weights of all layers are equally distributed among all NPUs. (b) With hybrid TP and PP, the layers are distributed among groups of accelerators. Within a group, the layer weights are distributed in tensor parallel fashion. (c) Full PP splits all layers across the NPUs. In all cases, once the model has been partitioned via TP/PP, DP is employed across the remaining NPUs in the platform.', use_container_width=True)
 
  st.markdown("""
Meeting SLO demands another reason to scale up the number of accelerators in the platform. However, there is a lack of clarity about the impact of the parallelism strategy. An optimal parallelization strategy may vary depending on the model and workload configuration. We also observe drastic contrast in how different parallelization strategies work during the prefill and decode stage. 
                  """)

  st.markdown("### Operator Fusion")
  st.markdown("""
  Techniques like FlashAttention/FLAT used to fuse multiple kernels together to speed up certain operations
    """)

with st.expander("Platform Overview"):
  st.image(platform_abstraction, caption='Abstraction for the smallest individual processing unit in GenZ.')
  st.markdown("""
Each NPU in GenZ has a certain amount of efficient compute cores, which can perform computations. 
Additionally, we incorporate a computing efficiency factor to account for inefficiencies caused by software and memory synchronization issues.
Each NPU provides access to external memories like an HBM/DDR Memory Bank. We also use an efficiency factor with the memory link for accurate memory access time. 
""")
  st.markdown("### Supported Hardware Platforms")
  st.markdown("""
Currently GenZ supports the most commonly used platforms from NVIDIA, Google, AMD, INTEL. We can also add more hardware support with details on [request](mailto:abambhaniya3@gatech.edu).      """)
  st.markdown("### Parallelism Schemes")
  st.markdown("""
  GenZ supports Tensor Parallelism (TP) and Pipeline Parallelism (PP) across large platforms with multiple NPUs.""")
    


with st.expander("GenZ Validation"):
  st.markdown("""
  Refer to Section IV.E of the [paper](https://arxiv.org/abs/2406.01698)
  """)

st.markdown("""
## Upcoming updates
Following updates are planned to come soon and stay tuned! Any contributions or feedback are highly welcome!

- [x] Add Expert parallelism and Sequence parallelism
- [ ] Support LoRA
- [x] Mamba model layers
- [ ] Add different kinds of quantization for weights/KV/activations

## Citation
"""
)


github_button = """
If a project helps you, please give it a star! [⭐️](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer)
"""
st.markdown(github_button, unsafe_allow_html=True)

st.markdown(
    """
If you use GenZ in your [paper](https://arxiv.org/abs/2406.01698), please cite:

```
@misc{bambhaniya2024demystifying,
      title={Demystifying Platform Requirements for Diverse LLM Inference Use Cases},
      author={Abhimanyu Bambhaniya and Ritik Raj and Geonhwa Jeong and Souvik Kundu and Sudarshan Srinivasan and Midhilesh Elavazhagan and Madhu Kumar and Tushar Krishna},
      year={2024},
      eprint={2406.01698},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```
"""
)

st.markdown(
    """
    </span>
    """,
    unsafe_allow_html=True
)