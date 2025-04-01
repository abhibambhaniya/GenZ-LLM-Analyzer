from pandas import DataFrame
from GenZ.analyse_model import simplify_df
from GenZ import Unit
def get_energy(
            df: DataFrame, 
            power: float = 1000,    ## Power in Watts
            power_breakdown: dict = {
                'Static': 30,
                'Compute':40,
                'Memory': 20,
                'Network': 10
            },
) -> float:
    '''
    Calculate the used energy of based on the utilization of various components
    df: DataFrame: Model DF with operator wise breakdown of LLM model
    energy: float|dict{float}: Energy consumption of various components in the platform
            Components: Static, Compute, Memory, Network
            
    Return: float: Used energy in kWh
    '''
    df = simplify_df(df)

    total_energy_used = 0
    static_power = power_breakdown['Static']*power
    compute_power = power_breakdown['Compute']*power
    memory_power = power_breakdown['Memory']*power
    network_power = power_breakdown['Network']*power
    unit = Unit() 
    for i in range(len(df)):
        total_energy_used += df.loc[i,f'Latency ({unit.unit_time})'] * (
                    (static_power) +
                    (compute_power*df.loc[i,'Compute Utilization']) +
                    (memory_power*df.loc[i,'Memory Utilization']) +
                    (network_power*df.loc[i,'Communication Utilization']))
        
    total_energy_used = total_energy_used / (1000 * 3600 * 1000)  ## Convert Energy from watts * msec to kWh

    return total_energy_used