from typing import Any, Dict

system_configs: Dict[str, Dict[str, Any]] = {
    'A100_40GB_GPU' : {'Flops': 312, 'Memory_size': 40, 'Memory_BW': 1600, 'ICN': 150 , 'real_values':True},
    'A100_80GB_GPU' : {'Flops': 312, 'Memory_size': 80, 'Memory_BW': 2039, 'ICN': 150 , 'real_values':True},
    'H100_GPU'  : {'Flops': 989, 'Memory_size': 80, 'Memory_BW': 3400, 'ICN': 450 , 'real_values':True},
    'GH200_GPU' : {'Flops': 1979, 'Memory_size': 144, 'Memory_BW': 4900, 'ICN': 450 , 'real_values':True},
    # https://resources.nvidia.com/en-us-blackwell-architecture?ncid=no-ncid
    "B100" : {'Flops': 3500, 'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900, 'ICN_LL':0.25, 'real_values':True},
    "GB200" : {'Flops': 4500, 'Memory_size': 192, 'Memory_BW': 8000, 'ICN': 900, 'ICN_LL':0.25, 'real_values':True},
    # https://www.nextplatform.com/2024/06/10/lots-of-questions-on-googles-trillium-tpu-v6-a-few-answers/ 
    "TPUv6" : {'Flops': 926, 'Memory_size': 32, 'Memory_BW': 1640, 'ICN': 100 , 'real_values':True},
    'TPUv5e' :  {'Flops': 197, 'Memory_size': 16, 'Memory_BW': 820, 'ICN': 50 , 'real_values':True},
    "TPUv5p" : {'Flops': 459, 'Memory_size': 95, 'Memory_BW': 2765, 'ICN': 450 , 'real_values':True},
    'TPUv4' : {'Flops': 275, 'Memory_size': 32, 'Memory_BW': 1228, 'ICN': 24 , 'real_values':True},
    # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
    'MI300X' : {'Flops': 1307, 'Memory_size': 192, 'Memory_BW': 5300, 'ICN': 400 , 'real_values':True},
    #https://www.google.com/search?client=safari&rls=en&q=amd+MI325X&ie=UTF-8&oe=UTF-8 
    "MI325X": {'Flops': 1307, 'Memory_size': 256, 'Memory_BW': 6000, 'ICN': 400 , 'real_values':True},
    # https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html
    'Gaudi3' : {'Flops': 1835, 'Memory_size': 128, 'Memory_BW': 3675, 'ICN': 300 , 'real_values':True},
}