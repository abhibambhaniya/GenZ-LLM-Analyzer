from GenZ.unit import Unit
import GenZ.operators as operators
from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
import numpy as np
import os
from GenZ.Models import OpType, ResidencyInfo

def get_attn_index(df:pd.DataFrame):
    ret = []
    for idx in range(len(df)):
        if 'Attend' in df.loc[idx, 'Op Type'] or 'Logit' in df.loc[idx, 'Op Type']:
            ret.append(idx)
    return ret

def get_summary_table(df:pd.DataFrame, unit = Unit(), model_characterstics:bool=False):

    attn_idx = get_attn_index(df)

    total_macs = 0
    total_data = 0
    kv_cache = 0
    total_weights = 0
    unused_weights = 0
    total_latencies = 0
    total_cycles = 0
    total_attn_latencies = 0
    total_linear_latencies = 0
    total_comm_latencies = 0

    multiplier = 1
    for i in range(len(df)):
        if df.loc[i,'Op Type'] == 'Repeat':
            multiplier *= df.loc[i,'Dimension']
        elif df.loc[i,'Op Type'] == 'EndRepeat':
            multiplier /= df.loc[i,'Dimension']
        else:
            total_macs += df.loc[i,f'Num ops ({unit.unit_flop})'] * multiplier
            total_data += (df.loc[i,f'Input_a ({unit.unit_mem})'] + df.loc[i,f'Input_w ({unit.unit_mem})'] + df.loc[i,f'Output ({unit.unit_mem})']) * multiplier
            if i in attn_idx:
                kv_cache += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier
            else:
                total_weights += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier
                if df.loc[i, f'Num ops ({unit.unit_flop})'] == 0:
                    unused_weights += df.loc[i,f'Input_w ({unit.unit_mem})']

            if model_characterstics == False:
                total_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                total_cycles += df.loc[i,'Cycles'] * multiplier
                if i in attn_idx:
                    total_attn_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                elif 'GEMM' in df.loc[i, 'Op Type']:
                    total_linear_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                elif 'Sync' in df.loc[i, 'Op Type']:
                    total_comm_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier

    max_memory_footprint = max([df.loc[i, f'Input_a ({unit.unit_mem})'] + df.loc[i, f'Input_w ({unit.unit_mem})'] + df.loc[i, f'Output ({unit.unit_mem})'] for i in range(len(df))])


    ret = {
            f'MACs ({unit.unit_flop})': [total_macs],
            f'Total Data ({unit.unit_mem})': [total_data],
            f'Total Weights ({unit.unit_mem})': [total_weights],
            f'Unused Weights ({unit.unit_mem})': [unused_weights],
            f'KV Cache ({unit.unit_mem})': [kv_cache],
            f'On-chip Memory Footprint ({unit.unit_mem})': [max_memory_footprint],
        }
    if model_characterstics == False:
        ret.update({
            f'Latency ({unit.unit_time})': [total_latencies],
            'Cycles': [total_cycles],
            f'Attn Latency ({unit.unit_time})': [total_attn_latencies],
            f'Linear Latency ({unit.unit_time})': [total_linear_latencies],
            f'Comm Latency ({unit.unit_time})': [total_comm_latencies]
        })


    return pd.DataFrame.from_dict(ret)

def analysis_model(model_dims, system=None, unit=Unit(), densities = None,intermediate_on_chip=False,
                    beam_size=1, beam_merge=False, model_characterstics=False):
    roofline_list = []
    if densities is None:
        densities = np.ones((len(model_dims), 3), dtype=float)
    for i, (dim, density) in enumerate(zip(model_dims, densities)):

        op_type = op_type_dicts[dim[-1]]
        operators_residency = dim[-2]
        operator = getattr(operators, op_type)
        if beam_merge and (dim[-1] == OpType.Logit_BM_PREFILL or dim[-1] == OpType.Attend_BM_PREFILL):
            dim[0] /= beam_size
        operator_instance = operator(dim=dim, density=density)
        # print(density[0],density[1],density[2])
        if (intermediate_on_chip):
            if(op_type == 'Logit'):
                operator_instance.set_mem_pin(output='on')
            elif(op_type == 'Attend'):
                operator_instance.set_mem_pin(input_a='on')

        if operators_residency == ResidencyInfo.A_onchip:
            operator_instance.set_mem_pin(input_a='on')
        elif operators_residency == ResidencyInfo.B_onchip:
            operator_instance.set_mem_pin(input_b='on')
        elif operators_residency == ResidencyInfo.C_onchip:
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.AB_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(input_b='on')
        elif operators_residency == ResidencyInfo.AC_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.BC_onchip:
            operator_instance.set_mem_pin(input_b='on')
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.All_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(input_b='on')
            operator_instance.set_mem_pin(output='on')

        if model_characterstics:
            roofline = operator_instance.get_model_characterstics(system=system, unit=unit)
        else:
            roofline = operator_instance.get_roofline(system=system, unit=unit)

        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)

    return df


def get_model_df(model, system=System(), unit=Unit(), batch_size=1, data_path="/tmp/genz/data", intermediate_on_chip=False,
                    beam_size=1, beam_merge=False, model_characterstics=False):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model)
    density_file = os.path.join(sparsity_file_path, model)
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size
    model_defs = np.append(batch_sizes, model_defs, axis=1)
    def verify_repeat_pairs(model_defs):
        pairs = []
        stack = []
        for idx, row in enumerate(model_defs):
            Repeat_id = row[2]
            if row[-1] == OpType.REPEAT:
                stack.append((idx, Repeat_id))
            elif row[-1] == OpType.ENDREPEAT:
                if stack and stack[-1][1] == Repeat_id:
                    start_idx, _ = stack.pop()
                    pairs.append((start_idx, idx))
                else:
                    raise ValueError(f"Unmatched Endrepeat found or ID mismatch:{Repeat_id}")

        if stack:
            raise ValueError(f"Unmatched Repeat found: {stack[-1]}")

        return pairs
    pairs = verify_repeat_pairs(model_defs)

    new_model_defs = []
    for layer in model_defs:
        if layer[-1] == OpType.EINSUM:
            new_layer = [int(x) if isinstance(x, (int, float)) else x for x in layer]
            new_model_defs.append(new_layer)
        else:
            new_model_defs.append(layer.astype(int))

    densities = np.ones((len(model_defs), 3), dtype=float)

    return analysis_model(new_model_defs, system, unit, densities, intermediate_on_chip, beam_size, beam_merge, model_characterstics)

