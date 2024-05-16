from collections import OrderedDict
from typing import Optional
from src.unit import Unit
import warnings
from src.system import System

## Hy00, Hy01, Hy10, Hy11,   A100-40GB, A100-80GB, H100, TPUv5e, TPUv4
system_name_list = {
    'LC,LM':(0,0,0), 'LC,HM':(0,1,0), 'HC,LM':(1,0,0), 'HC,HM':(1,1,0), 
    'A100-40GB':(2,2,2), 'A100-80GB':(3,3,3), 'H100':(4,4,4), 'GH200':(7,7,7),
    'TPUv5e':(5,5,5), 'TPUv4':(6,6,6),  'MI300X':(8,8,8), 'gaudi3':(9,9,9), 
    'groq':(10,10,10),   ## 8 Chips
    }

FLOPS_dict       = [  64,    512,    312,   312,   700, 197, 275, 1979,  1307,  1600, 750 ]      ## TFLOPS
Memory_size_dict = [  16,    128,     40,     80,    80,  16,  32,  144,   192,   144, 1.76/8 ]   ## GB
Memory_bw_dict   = [1200,   4000,   1600,   2039,  2300, 820, 1200, 4900, 5300,  3675, 8*1024  ]   ## GBps
ICN_BW_dict      = [  50,    450,    150,    150,   350,  50,  50,   450,   400,  300,  25 , 1e9]  ## GBps   
offload_bw       = 128    ## GBps

Cost_per_chip     =[1.54,  2.5,    1.7 ,     5,    0.79,    1.69,   4.09,      1.2,    3.22,       7]


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

def get_inference_system(system_name='A100-40GB', bits='bf16'):
    ################################################################################################## # 
    ### System Declaration
    ################################################################################################## # 
    if isinstance(system_name, str):
        system_config = system_name_list.get(system_name, None) 
        if system_config is None:
            raise NameError(f'system {system_name} should be in {system_name_list.keys()}')

        NUM_FLOPS = FLOPS_dict[system_config[0]] 
        OFFCHIP_MEM_BW = Memory_bw_dict[system_config[1]] 
        per_chip_memory = Memory_size_dict[system_config[1]]
        C2C_BW = ICN_BW_dict[system_config[2]]
        C2C_LL = 1.9
    elif isinstance(system_name, dict):
        if system_name.get('real_values',False) == True:
            NUM_FLOPS = system_name.get('Flops', 320)
            OFFCHIP_MEM_BW = system_name.get('Memory_BW',40) 
            per_chip_memory = system_name.get('Memory_size',2000)
            C2C_BW = system_name.get('ICN',150)
            C2C_LL = system_name.get('ICN_LL',2)

        else:
            _unused_keys = [ k for k in system_name.keys() if k not in ['Flops', 'Memory', 'ICN', 'real_values']]
            if len(_unused_keys) > 0:
                warnings.warn(f"Following keys of system_name are not used: {_unused_keys}")

            _missing_keys = [ k for k in ['Flops', 'Memory', 'ICN'] if k not in system_name.keys()]
            if len(_missing_keys) > 0:
                warnings.warn(f"Following keys of system_name are missing: {_missing_keys}") 

            NUM_FLOPS = FLOPS_dict[system_name.get('Flops', 2)] 
            OFFCHIP_MEM_BW = Memory_bw_dict[system_name.get('Memory',2)] 
            per_chip_memory = Memory_size_dict[system_name.get('Memory',2)]
            C2C_BW = ICN_BW_dict[system_name.get('ICN',2)]
            C2C_LL = 1.9
    else:
        raise TypeError('System should be weight str or dict with Flops,Memory, ICN values') 
    
    return System(unit,frequency=1000 , flops=NUM_FLOPS, off_chip_mem_size=(per_chip_memory*1024),
                     offchip_mem_bw=OFFCHIP_MEM_BW, bits=bits, external_mem_bw=offload_bw, interchip_mem_bw=C2C_BW, interchip_link_latency=C2C_LL)