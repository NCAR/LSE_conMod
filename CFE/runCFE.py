# Helper file to couple CFE to LSE similar to how the HBV is running 
# Similar structure to ../HBV/HBV.py
# Created by Rhys Pulling

import pickle, os, sys, re, json, ast
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gamma 
import bmi_cfe

# CODE FROM HBV.PY
def UH_gamma(a, b, lenF=10):
    """
    Generates a unit hydrograph (UH) using a gamma distribution for 1D inputs.

    Parameters:
        a: float
            Shape parameter of the gamma distribution.
        b: float
            Scale parameter of the gamma distribution.
        lenF: int
            Length of the unit hydrograph.

    Returns:
        np.ndarray
            Unit hydrograph (1D array of weights summing to 1).
    """
    lenF = max(lenF, 1)  # Ensure lenF is at least 1
    t = np.arange(0.5, lenF, 1)  # Time steps for the UH
    a = max(a, 0.1)  # Enforce a minimum value for `a`
    b = max(b, 0.5)  # Enforce a minimum value for `b`

    denom = gamma(a) * (b ** a)
    mid = t ** (a - 1)
    right = np.exp(-t / b)
    w = (1 / denom) * mid * right
    w = w / w.sum()  # Normalize weights to sum to 1

    return w

# CODE FROM HBV.PY
def UH_conv(x, UH):
    """
    Convolves an input time series with a unit hydrograph (UH).

    Parameters:
        x: np.ndarray
            Input time series (1D array).
        UH: np.ndarray
            Unit hydrograph (1D array).

    Returns:
        np.ndarray
            Routed output time series (1D array).
    """
    # Flip the UH for proper convolution
    UH_flipped = UH[::-1]

    # Perform the convolution
    Qr = np.convolve(x, UH_flipped, mode='full')

    # Trim to the original input length
    Qr = Qr[:len(x)]

    return Qr

# use the gamma functions to route the runoff
# CODE FROM HBV.PY
def route_flow(routing_params, Qsim, routing=True):
    if routing:
        # Generate unit hydrograph
        UH = UH_gamma(routing_params['rout_a'], routing_params['rout_b'], lenF=15)

        # Perform routing
        Qsrout = UH_conv(Qsim, UH)
        
    else:
        Qsrout = Qsim
    
    return Qsrout

# NEED TO SET UP TO GRAB PARAMS ACCURATELY
def load_CFE_param(infile_param, area, depth):
    # read CFE parameters from a pickle file

    try:
        df = pd.read_pickle(infile_param)
    except:
        df = pd.read_csv(infile_param)
    
    param1 = ['soil_params.bb', 'soil_params.satdk', 'soil_params.satpsi', 'soil_params.slop', 'soil_params.smcmax', 'soil_params.wltsmc', 'max_gw_storage', 
              'Cgw', 'expon', 'K_nash', 'K_lf', 'nash_storage', 'a_Xinanjiang_inflection_point', 'b_Xinanjiang_shape_param', 'x_Xinanjiang_shape_param']
    param2 = ['rout_a', 'rout_b'] 

    param1_dict = {}
    for p in param1:
        param1_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]
    
    ### grab the non-calibrated params ##
    param1_dict["area"] = area
    param1_dict["soil_params.D"] = depth
    
    # get the alpha_fc + giuh_ordinate values
    df_basin = pd.read_csv("/home/m10870071/lse/CFE/basin_params.csv")
    
    file_path = Path(infile_param)
    gageID_raw = file_path.parts[-3]
    if len(gageID_raw) < 8:
        gageID = f"0{gageID_raw}"
    else:
        gageID = gageID_raw
    param1_dict["gageID"] = gageID
    
    df_basin['giuh_ordinates'] = df_basin['giuh_ordinates'].apply(normalize_giuh) # not sure if needed anymore
        
    row = df_basin.loc[df_basin['basin_id'] == gageID, ['alpha_fc', 'giuh_ordinates']]
    if not row.empty:
        param1_dict["alpha_fc"] = row['alpha_fc'].values[0]
        param1_dict["giuh_ordinates"] = row['giuh_ordinates'].values[0]
        if len(param1_dict["giuh_ordinates"]) == 0:
            raise ValueError(f"No GIUH ordinates found for basin_id {gageID}")
    else:
        raise ValueError(f"basin_id {gageID} not found in CSV")
        
    param2_dict = {}
    for p in param2:
        param2_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    return param1_dict, param2_dict

# Ensure giuh_ordinates column is always a list of floats
def normalize_giuh(x):
    # If it's already a list, leave it
    if isinstance(x, list):
        return [float(i) for i in x]
    # If it's a string, split by commas
    elif isinstance(x, str):
        return [float(i) for i in x.split(",") if i.strip() != ""]
    # If it's a NumPy array, convert to list
    elif isinstance(x, np.ndarray):
        return [float(i) for i in x.tolist()]
    # If it's a scalar (single number), wrap in list
    elif np.isscalar(x):
        return [float(x)]
    # Fallback: empty list
    return []

# Store the cfe outputs to a pickle file for future use
def save_CFE_output(filename, data, overwrite=True):
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

# Grab the cfe outputs from a pickle file
def load_CFE_output(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Function to replace specific values in the
# template configuration
def replace_placeholders(obj, param_dict):
    if isinstance(obj, dict):
        return {k: replace_placeholders(v, param_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_placeholders(item, param_dict) for item in obj]
    elif isinstance(obj, str):
        # Check if the entire string is a single placeholder
        match = re.fullmatch(r"{{(.*?)}}", obj)
        if match:
            key = match.group(1)
            return param_dict.get(key, obj)  # preserve original type if available
        else:
            # Replace placeholders inside longer strings, result will be string
            def repl(m):
                key = m.group(1)
                return str(param_dict.get(key, m.group(0)))
            return re.sub(r"{{(.*?)}}", repl, obj)
    else:
        return obj

# replace the template with param values
def generate_config(param_dict, file_paramset, config_template):
    # get the template for the configuration file
    template_path = config_template
    
    # get the file path and add to config_file directory    
    file_path = Path(file_paramset) # f'{outpath_parent}/{gageID}/param_sets/paramset_iter{iter}_trial{tarparam}.pkl'

    # Extract parts
    gageID = file_path.parts[-3]  # gageID is two levels up from the file
    outpath_parent = file_path.parents[2]  # outpath_parent is three levels up
    filename = file_path.stem  # 'paramset_iter{iter}_trial{tarparam}'

    # Extract iter and tarparam using string methods
    _, iter_part, trial_part = filename.split('_')
    iter_val = iter_part.replace("iter", "")
    tarparam_val = trial_part.replace("trial", "")
    
    # Set up output path
    output_dir = f'{outpath_parent}/{gageID}/config_files/'  
    output_filename = f'config_{iter_val}_{tarparam_val}.json'
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # print("Config file found at :", output_path)

    # Load template JSON as a dict
    with open(template_path, 'r') as f:
        template_json = json.load(f)

    # Recursively replace placeholders
    final_json = replace_placeholders(template_json, param_dict)

    # Write final config
    with open(output_path, 'w') as f:
        json.dump(final_json, f, indent=2)

    return output_path

# convert the m/hr to mm/day
# CURRENTLY JUST DOING M/DAY
def convert_to_daily(hourly):
    n = len(hourly) - (len(hourly) % 24)
    hourly_trimmed = hourly[:n]
    daily = hourly_trimmed.reshape(-1, 24)
    daily = daily.mean(axis=1) # may need to convert to sum
    return daily

def run_CFE_OneBasin_OneParamSet(Ptar, Ttar, PETtar, AREAtar, DEPTHtar, file_paramset, outfile, config_template):
    '''
    Use the Basic Model Interface to interact with the CFE to 
    run it with the given parameters. 
    '''
    
    CFEparam_dict, routingparam_dict = load_CFE_param(file_paramset, AREAtar, DEPTHtar)
    CFEconfig_file = generate_config(CFEparam_dict, file_paramset, config_template) 
        
    ##### run CFE using BMI #####

    # initialize
    cfe_instance = bmi_cfe.BMI_CFE(CFEconfig_file)
    cfe_instance.initialize()

    if os.path.isfile(outfile):
        print("CFE output file exists: ", outfile)
        print("Skip CFE Modeling")
        
        ds = xr.open_dataset(cfe_instance.forcing_file) # aorc forcings are .nc files
        df_forcing = ds.to_dataframe().reset_index()
        
        qobs = df_forcing["qobs_mm_per_hour"]
        return qobs # compare to AORC out flow

    # initialize => update for each time step
    else:
        outputs = cfe_instance.get_output_var_names()
        output_lists = {output:[] for output in outputs}
        output_lists["runoff_depth_mm"] = []
        
        forcingFile = True # use the forcing file specified in the CFE
        
        if forcingFile:
            ds = xr.open_dataset(cfe_instance.forcing_file) # aorc forcings are .nc files
            df_forcing = ds.to_dataframe().reset_index()
            
            for n, precip in enumerate(df_forcing['APCP_surface']):
                # if n >= len(PETtar):
                #     break
                # pet = PETtar[n].item()
                
                precip_m = precip / 1000.0
                cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_m)
                #cfe_instance.set_value('water_potential_evaporation_flux', pet)
                cfe_instance.update()

                for output in outputs:
                    if output == "land_surface_water__runoff_depth":
                        output_lists["runoff_depth_mm"].append(cfe_instance.get_value(output) * 1000.0) # convert to mm/hr
                        output_lists[output].append(cfe_instance.get_value(output)) 
                    else:
                        output_lists[output].append(cfe_instance.get_value(output))

#         else:
#             n_steps = len(Ptar)

#             for n in range(n_steps):
#                 precip = Ptar[n].item()
#                 # temp = Ttar[n].item()
#                 # pet = PETtar[n].item()

#                 cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip) # assuming precip is m/hr
#                 # cfe_instance.set_value('water_potential_evaporation_flux', pet)
#                 cfe_instance.update()
#                 for output in outputs:
#                     output_lists[output].append(cfe_instance.get_value(output))
            
        # finalize
        cfe_instance.finalize(print_mass_balance=False)
        
        # convert values to np arrays
        #output_lists["flow_sim"] = convert_to_daily(np.array(output_lists["land_surface_water__runoff_depth"])) # convert to mm/day
        # output_lists["runoff_depth_mm"] = route_flow(routingparam_dict, output_lists["runoff_depth_mm"]) # route values
        output_arrays = {key: np.array(value) for key, value in output_lists.items()} # convert to numpy array for speed
                
        print("Saving output file to ", outfile)        
        save_CFE_output(outfile, output_arrays)
        
        qobs = df_forcing["qobs_mm_per_hour"]
        return qobs # compare to AORC out flow