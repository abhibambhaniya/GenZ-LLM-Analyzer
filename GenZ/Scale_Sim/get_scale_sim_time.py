
import warnings
import os
import subprocess
import yaml
from GenZ.system import System
from GenZ.unit import Unit
from GenZ.operator_base import Operator
import numpy as np
import contextlib
import io
import re
import pandas as pd

def get_scale_sim_time(op:Operator,  system: System):
    # Get the time taken by Scale-sim to simulate the system
    # system: System object
    # returns: time taken by Scale-sim to simulate the system
    # Note: This function is a wrapper around the Scale-sim simulator
    # and requires the Scale-sim simulator to be installed on the system
    op_type = op.get_op_type(op.dim)
    op_dim = op.dim[:op.get_effective_dim_len()]
    runtime_data = pd.read_csv('/Users/abambhaniya3/GenZ/GenZ_paper_charts/v2/rebbutal/LLM inference bench results/Scale_sim_runtimes-GenZ.csv')
    if op_type == 'GEMM':
        outer, weightdim, inputdim, contract = op_dim
        M = inputdim
        N = weightdim
        K = contract    ## K is the contract dimension
        B = outer       ## B is the outer dimension
        # return left, upper, contract, outer
        row = runtime_data[(runtime_data['B'] == B) &
                   (runtime_data['M'] == M) &
                   (runtime_data['N'] == N) &
                   (runtime_data['K'] == K)]
        if not row.empty:
            if system.mxu_shape == (256, 256):
                return row['256'].values[0] / system.frequency
            elif system.mxu_shape == (4, 128, 128) or system.mxu_shape == (16, 64, 64):
                return row['128'].values[0] / system.frequency
        else:
            raise ValueError("No matching runtime data found: B={}, M={}, N={}, K={}".format(B, M, N, K))

    elif op_type == 'Logit':
        B, H, M, N, D, Hkv = op_dim
        
        row = runtime_data[(runtime_data['B'] == B*H) &
            (runtime_data['M'] == M) &
            (runtime_data['N'] == N) &
            (runtime_data['K'] == D)]
        if not row.empty:
            if system.mxu_shape == (256, 256):
                return row['256'].values[0] / system.frequency
            elif system.mxu_shape == (4, 128, 128):
                return row['128'].values[0] / system.frequency
            elif system.mxu_shape == (16, 64, 64):
                return op.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/(system.op_per_sec / 20)
        else:
            raise ValueError("No matching runtime data found: B={}, M={}, N={}, D={}".format(B*H, M, N, D))

    elif op_type == 'Attend':
        B, H, M, N, D, Hkv = op_dim
        row = runtime_data[(runtime_data['B'] == B*H) &
            (runtime_data['M'] == M) &
            (runtime_data['N'] == D) &
            (runtime_data['K'] == N)]
        if not row.empty:
            if system.mxu_shape == (256, 256):
                return row['256'].values[0] / system.frequency
            elif system.mxu_shape == (4, 128, 128):
                return row['128'].values[0] / system.frequency
            elif system.mxu_shape == (16, 64, 64):
                return op.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/(system.op_per_sec / 20)
        else:
            raise ValueError("No matching runtime data found: B={}, M={}, N={}, D={}".format(B*H, M, N, D))

    
    return op.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec