from collections import OrderedDict
from typing import Optional
from GenZ.unit import Unit
import warnings
from GenZ.system import System

from Systems.system_configs import system_configs

offload_bw = 128 

unit = Unit()

class ModdelingOutput(dict):
    Latency: float = None
    Throughput: float = None
    Runtime_breakdown: Optional[list[float]] = None 
    is_offload: Optional[bool] = False

def get_offload_system(system, total_memory_req, debug):

    
    total_device_memory = unit.raw_to_unit(system.off_chip_mem_size, type='M')/1024 ## GB

    memory_offloaded = total_memory_req - total_device_memory

    if debug:
        print(f'Total Memory Req:{total_memory_req}, Total device mem:{total_device_memory}, Mem Offload:{memory_offloaded}')

    ###############################################################################################
    ### Memory Time = max(min(size_required,size_hbm)/BW_hbm, (size_required-size_hbm)/BW_offload)
    ###############################################################################################

    if memory_offloaded > 0: 
        new_offchip_BW = (total_memory_req ) / max(min(total_memory_req,total_device_memory)/unit.raw_to_unit(system.offchip_mem_bw, type='BW'), memory_offloaded/offload_bw) 
        system.set_offchip_mem_bw(new_offchip_BW)
        if debug:
            print(f'New BW:{new_offchip_BW}')
    return system

def get_inference_system(system_name='A100_40GB_GPU', bits='bf16', ceff=1, meff=1):
    ################################################################################################## # 
    ### System Declaration
    ################################################################################################## # 
    if isinstance(system_name, str):
        if system_name in system_configs:
            system_name = system_configs[system_name]
        else:
            raise ValueError(f'System mentioned:{system_name} not present in predefined systems. Please use systems from Systems/system_configs')
    if isinstance(system_name, dict):
        if system_name.get('real_values',False) == True:
            NUM_FLOPS = system_name.get('Flops', 320)
            OFFCHIP_MEM_BW = system_name.get('Memory_BW',40) 
            per_chip_memory = system_name.get('Memory_size',2000)
            C2C_BW = system_name.get('ICN',150)
            C2C_LL = system_name.get('ICN_LL',2)
    elif isinstance(system_name, System): 
        return system_name
    else:
        raise TypeError('System should be weight str or dict with Flops,Memory, ICN values') 
    
    return System(unit,frequency=1000 , flops=NUM_FLOPS, off_chip_mem_size=(per_chip_memory*1024), compute_efficiency=ceff, memory_efficiency=meff,
                     offchip_mem_bw=OFFCHIP_MEM_BW, bits=bits, external_mem_bw=offload_bw, interchip_mem_bw=C2C_BW, interchip_link_latency=C2C_LL)