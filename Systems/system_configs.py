from typing import Any, Dict

system_configs: Dict[str, Dict[str, Any]] = {
'A100_40GB_GPU' : {'Flops': 312, 'Memory_size': 40, 'Memory_BW': 1600, 'ICN': 150 , 'real_values':True},
'A100_80GB_GPU' : {'Flops': 312, 'Memory_size': 80, 'Memory_BW': 2039, 'ICN': 150 , 'real_values':True},
'H100_GPU'  : {'Flops': 989, 'Memory_size': 80, 'Memory_BW': 3400, 'ICN': 450 , 'real_values':True},
'GH200_GPU' : {'Flops': 1979, 'Memory_size': 144, 'Memory_BW': 4900, 'ICN': 450 , 'real_values':True},
'TPUv5e' :  {'Flops': 197, 'Memory_size': 16, 'Memory_BW': 820, 'ICN': 50 , 'real_values':True}, 
'TPUv4' : {'Flops': 275, 'Memory_size': 32, 'Memory_BW': 1200, 'ICN': 50 , 'real_values':True}, 
'MI300X' : {'Flops': 1307, 'Memory_size': 192, 'Memory_BW': 5300, 'ICN': 400 , 'real_values':True},  
'Gaudi3' : {'Flops': 1600, 'Memory_size': 144, 'Memory_BW': 3675, 'ICN': 300 , 'real_values':True}, 
}