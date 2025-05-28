from src import BudSimHardwares as bh

budHardware = bh.BudHardwares()
print(budHardware.add_hardware("BUD_80GB_GPU", {"Flops": 312, "Memory_size": 80, "Memory_BW": 2039, "ICN": 150 , "real_values":True}))
print(budHardware.list_hardwares())