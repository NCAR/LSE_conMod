# Function to run a snow routine for the CFE model
# Based on the SHM setup in the NeuralHydrology repository
# Specifically looks at the forward pass in the model
# https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.modelzoo.shm.html 

import numpy as np

# Compares temp to the threshold and converts to either snow or 
# precip, which can then be passed as input to CFE model
def shm_snow_step(precip_mm, temp_C, snow_storage, T_thresh=0.0, DDF=3.0, delta_K=1.0):
    """
    One step of the SHM snow routine. Compare temperature and convert
    to snow or runoff depending on threshold. 
    
    Parameters:
        precip_mm : precipitation input [mm] this time step
        temp_C    : temperature [°C] this time step
        snow_storage : snowpack at beginning of step [mm]
        T_thresh  : rain/snow threshold [°C]
        DDF       : degree-day factor [mm/°C/day]
        delta_K   : scaling factor for DDF (ex. delta_K = 24.0 scales to hourly time-step)
    
    Returns
        melt : snowmelt released [mm]
        rain : rainfall passed directly [mm]
        new_snow_storage : updated snowpack [mm]
    """
    if temp_C <= T_thresh:
        # all precip stored as snow
        new_snow_storage = snow_storage + precip_mm
        melt = 0.0
        rain = 0.0
    else:
        # precip falls as rain
        rain = precip_mm
        # melt potential (scaled by delta_K)
        melt_potential = (DDF / delta_K) * (temp_C - T_thresh)
        melt = min(snow_storage, melt_potential)
        new_snow_storage = snow_storage - melt
    
    return melt, rain, new_snow_storage

