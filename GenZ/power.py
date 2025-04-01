from pandas import DataFrame
from GenZ.analyse_model import simplify_df
from GenZ import Unit
def get_power(
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
    Calculate the used power of based on the utilization of various components
    df: DataFrame: Model DF with operator wise breakdown of LLM model
    power: float|dict{float}: Power consumption of various components in the platform
            Components: Static, Compute, Memory, Network
            
    Return: float: Used power in kWh
    '''
    df = simplify_df(df)

    used_power = power_breakdown['Static']*power
    compute_power = power_breakdown['Compute']*power
    memory_power = power_breakdown['Memory']*power
    network_power = power_breakdown['Network']*power
    unit = Unit() 
    for i in range(len(df)):
        used_power += df.loc[i,f'Latency ({unit.unit_time})'] * (
                    (compute_power*df.loc[i,'Compute Utilization']) + 
                    (memory_power*df.loc[i,'Memory Utilization']) +
                    (network_power*df.loc[i,'Communication Utilization']))
        
    used_power = used_power / (1000 * 3600 * 1000)  ## Convert power from watts * msec to kWh

    return used_power