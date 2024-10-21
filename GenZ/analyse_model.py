from GenZ.unit import Unit
from GenZ.operators import *
import GenZ.operators as operators
from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
import os
from GenZ.Models.get_language_model import OpType, ResidencyInfo

def get_attn_index(df):
    ret = []
    for idx in range(len(df)):
        if 'Attend' in df.loc[idx, 'Op Type'] or 'Logit' in df.loc[idx, 'Op Type']:
            ret.append(idx)
    return ret

def get_summary_table(df,system,unit = Unit(),sparse_pe_support=1, model_characterstics=False):
    if model_characterstics == False:
        total_cycles = np.sum(df['Cycles'])
        total_latencies = np.sum(df['Latency (msec)'])

    attn_idx = get_attn_index(df)
    total_parameters = np.sum(df['Input_w (MB)']) - sum([df.loc[i, 'Input_w (MB)'] for i in attn_idx])
    total_data = np.sum(df['Input_a (MB)'] + df['Input_w (MB)'] + df['Output (MB)'])
    total_MACS = np.sum(df['Num ops (MFLOP)'])
    # total_weights = np.sum(df['Input_w (MB)'])
    total_weights = 0;
    for i in range(len(df)):
        if ('Logit' not in df.loc[i, 'Op Type']  and 'Attend' not in df.loc[i, 'Op Type']):
            total_weights = total_weights + df.loc[i,'Input_w (MB)']
    max_memory_footprint = max([df.loc[i, 'Input_a (MB)'] + df.loc[i, 'Input_w (MB)'] + df.loc[i, 'Output (MB)'] for i in range(len(df))])


    ret = {
            f'MACs ({unit.unit_flop})': [total_MACS],
            f'Total Data ({unit.unit_mem})': [total_data],
            f'Total Weights ({unit.unit_mem})': [total_weights],
            f'Parameters  ({unit.unit_mem})': [total_parameters],
            f'On-chip Memory Footprint ({unit.unit_mem})': [max_memory_footprint],
        }
    if model_characterstics == False:
        ret.update({
            f'Latency ({unit.unit_time})': [total_latencies],
            'Cycles': [total_cycles],
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
        if beam_merge and (dim[-1] == 9 or dim[-1] == 10):
            dim[0] /= beam_size
        operator_instance = operator(dim=dim, density=density)
        # print(density[0],density[1],density[2])
        if (intermediate_on_chip):
            if(op_type == 'Logit' or op_type == 'Logit_MQA'):
                operator_instance.set_mem_pin(output='on')
            elif(op_type == 'Attend'or op_type == 'Attend_MQA'):
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

    # pd.set_option("precision", 3)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)

    return df


def get_model_df(model, system, unit=Unit(), batch_size=1, data_path="/tmp/genz/data", intermediate_on_chip=False,
                    beam_size=1, beam_merge=False, model_characterstics=False):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model)
    density_file = os.path.join(sparsity_file_path, model)
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size
    model_defs = np.append(batch_sizes, model_defs, axis=1)
    
    new_model_defs = []
    for layer in model_defs:
        if layer[-1] != OpType.EINSUM:
            new_model_defs.append(layer.astype(int))
        else:
            new_layer = [int(x) if isinstance(x, (int, float)) else x for x in layer]
            new_model_defs.append(new_layer)

    densities = np.ones((len(model_defs), 3), dtype=float)

    model_df  = analysis_model(new_model_defs, system, unit, densities, intermediate_on_chip, beam_size, beam_merge, model_characterstics)
    return model_df