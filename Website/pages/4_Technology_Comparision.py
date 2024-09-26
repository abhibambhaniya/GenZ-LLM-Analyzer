# Import necessary libraries
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from GenZ import decode_moddeling, prefill_moddeling, get_configs
import pandas as pd
from tqdm import tqdm
import time
import copy


from Systems.system_configs import system_configs


st.set_page_config(
    page_title="Technolgy Comparisons",
    page_icon="🔬",

    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues',
        'Report a bug': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/issues",
        'About': "https://github.com/abhibambhaniya/GenZ-LLM-Analyzer/blob/main/README.md"
    }
)

st.sidebar.title("Technolgy Comparisons")


st.title("Coming Soon.")


# def main():
    
#     st.write(f"")

# if __name__ == "__main__":
#     main()
