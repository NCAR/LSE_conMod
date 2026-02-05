# Helper file to couple CFE to LSE similar to how the HBV is running 
# Similar structure to ../HBV/HBV.py. Has a lot of the same code with
# some minor adjustments to help run the CFE.
# Created by Rhys Pulling

import pickle, os, sys, re, json, ast
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gamma 
import bmi_cfe
from runSnow import shm_snow_step

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

# CODE FROM HBV.PY
def route_flow(routing_params, Qsim, routing=True):
    '''
    Uses the gamma functions to route the runoff from the 
    CFE model. 
    
    Parameters:
        routing_params: The shape and scale of the routing function
        Qsim: The output values from the CFE model
        
    Returns:
        np.ndarray
            Routed output time series (1D array).
    '''
    if routing:
        # Generate unit hydrograph
        UH = UH_gamma(routing_params['gamma_shape'], routing_params['gamma_scale'], lenF=15)

        # Perform routing
        Qsrout = UH_conv(Qsim, UH)
        
    else:
        Qsrout = Qsim
    
    return Qsrout

def load_CFE_param(infile_param, area, depth):
    '''
    Grabs the generated parameters from the pickle file and 
    stores them in dictionaries to be used later on. 
    
    Parameters:
        infile_param: The file where the parameters were stored
        area: The area of the basin
        depth: The soil depth of the basin
        
    Returns: 
        Three Python dictionaries containing parameters
            paramdict1: CFE model parameters
            paramdict2: Routing parameters
            paramdict3: Snow Routine parameters
    '''

    try:
        df = pd.read_pickle(infile_param)
    except:
        df = pd.read_csv(infile_param)
    
    param1 = ['soil_params.bb', 'soil_params.satdk', 'soil_params.satpsi', 'soil_params.slop', 'soil_params.smcmax', 'soil_params.wltsmc', 'max_gw_storage', 
              'Cgw', 'expon', 'K_nash', 'K_lf', 'nash_storage', 'a_Xinanjiang_inflection_point', 'b_Xinanjiang_shape_param', 'x_Xinanjiang_shape_param']
    param2 = ['gamma_shape', 'gamma_scale']
    param3 = ['DDF', 'T_thresh']

    param1_dict = {}
    for p in param1:
        param1_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]
    
    ### grab the non-calibrated params ##
    param1_dict["area"] = area
    param1_dict["soil_params.D"] = depth
    
    # get the alpha_fc + giuh_ordinate values
    df_basin = pd.read_csv("/home/m10870071/lse/HBV_LSE/data/basin_params.csv")
    
    file_path = Path(infile_param)
    gageID_raw = file_path.parts[-3]
    if len(gageID_raw) < 8:
        gageID = f"0{gageID_raw}"
    else:
        gageID = gageID_raw
    param1_dict["gageID"] = gageID
    
    df_basin['giuh_ordinates'] = df_basin['giuh_ordinates'].apply(normalize_giuh)
        
    row = df_basin.loc[df_basin['basin_id'] == gageID, ['alpha_fc', 'giuh_ordinates']]
    if not row.empty:
        param1_dict["alpha_fc"] = row['alpha_fc'].values[0]
        param1_dict["giuh_ordinates"] = row['giuh_ordinates'].values[0]
        if len(param1_dict["giuh_ordinates"]) == 0:
            raise ValueError(f"No GIUH ordinates found for basin_id {gageID}")
    # set up default values (for now)
    else:
        param1_dict["alpha_fc"] = 0.33
        param1_dict["giuh_ordinates"] = [0.35, 0.24, 0.17, 0.10, 0.06]
        # raise ValueError(f"basin_id {gageID} not found in CSV")
        
    param2_dict = {}
    for p in param2:
        param2_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]
        
    param3_dict = {}
    for p in param3:
        param3_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    return param1_dict, param2_dict, param3_dict

def normalize_giuh(x):
    '''
    Ensures the GIUH Ordinates grabbed for the model are
    set up properly. This avoids any issues that arise 
    from trying to place them in the config file for the 
    CFE.
    
    Parameters:
        x: The GIUH Ordinates
        
    Returns:
        Python list
            The correct set up for the GIUH ordinates
    '''
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

def save_CFE_output(filename, data, overwrite=True):
    '''
    Stores the CFE data in the specified pickle file to be 
    grabbed later. 
    
    Parameters:
        filename: The path/file to store the data in
        data: The CFE output data
    '''
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

def load_CFE_output(filename):
    '''
    Loads the CFE data from the specified pickle file.
    
    Parameters: 
        filename: The path/file to grab the data from
        
    Returns:
        The data from the file.
    '''
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def replace_placeholders(obj, param_dict):
    '''
    Helper function for generating the config file. 
    Finds the specified parameter in the template and
    recursively replaces it with the given values.
    
    Parameters:
        obj: The config template
        param_dict: The dictionary with the parameters
    Returns:
        String (file path)
            The config file with the filled in parameters
    '''
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

def generate_config(param_dict, file_paramset, config_template):
    '''
    Generate the BMI configuration file to run the CFE with. Grabs
    the parameters from the dictionary and replaces them in the template
    file. Stores in the specified path.
    
    Parameters:
        param_dict: A dictionary holding the parameters for the model
        file_paramset: The path to the generated parameters for the CFE model
        config_template: The template for the 
    Returns:
        String (file path)
            The config file with the filled in parameters
    '''
    # get the template for the configuration file
    template_path = config_template
    
    # get the file path and add to config_file directory    
    file_path = Path(file_paramset) 

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

# I used this function to generate new AORC NetCDF files
# so not necessarily needed any more
def penman_monteith_pet_hourly(df, albedo=0.23, emissivity=0.98):
    """
    Compute hourly PET (mm/hr) using the ASCE Penman–Monteith equation.
    Only used for our AORC data. 

    Parameters:
        df: Pandas dataframe containing...
            TMP_2maboveground (°C)
            PRES_surface (Pa)
            DSWRF_surface (W/m2)
            DLWRF_surface (W/m2)
            SPFH_2maboveground (kg/kg)
            UGRD_10maboveground (m/s)
            VGRD_10maboveground (m/s)
       
    Returns:
        Pandas DF
            Includes the PET [mm/hr] in df as a new column
    """
   
    T = df['TMP_2maboveground']                    # °C
    P = df['PRES_surface'] / 1000.0                # convert Pa to kPa
    q = df['SPFH_2maboveground'] / 1000.0          # convert from g/g to kg/kg (its a ratio so unsure if even needed)

    sigma = 5.67e-8                            # Stefan–Boltzmann [W/m²K⁴]
    R_sw_down = 0.77 * df['DSWRF_surface']     # net incoming shortwave (W/m²)
    R_l_down = df['DLWRF_surface']            # incoming longwave (W/m²)
    R_l_up   = 0.96 * sigma * (T + 273.15)**4
    Rn_W = (R_sw_down + R_l_down - R_l_up)    # net radiation in W/m²
    Rn = Rn_W * 0.0036 
    
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))  # saturation VP (kPa):contentReference[oaicite:8]{index=8}
    ea = (q * P) / (0.622 + 0.378 * q)  # actual VP:contentReference[oaicite:9]{index=9}
    Δ = (4098 * es) / ((T + 237.3)**2)             # slope of vapor pressure curve (kPa/°C)
    γ = 0.000665 * P                               # psychrometric constant (kPa/°C)
    VPD = es - ea   # vapor pressure deficit (kPa)
    
    wind10 = np.sqrt(df['UGRD_10maboveground']**2 + df['VGRD_10maboveground']**2)
    u2 = wind10 * 0.748                           # wind at 2 m (m/s)
    C_n, C_d = 37.0, 0.24                         # short-grass hourly constants:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    ET_num = 0.408 * Δ * (Rn - 0.0) + γ * (C_n/(T+273.15)) * u2 * VPD
    ET_den = Δ + γ * (1 + C_d * u2)
    df['potential_evaporation'] = ET_num / ET_den
    
    return df

def run_CFE_OneBasin_OneParamSet(Ptar, Ttar, PETtar, AREAtar, DEPTHtar, file_paramset, outfile, config_template, runRouting=True, runSHM=True, runAraki=False, endIndex=52620):
    '''
    Use the Basic Model Interface to interact with the CFE to 
    run it with the given parameters. Currently runs on either
    AORC, NLDAS, or Yalan Data depending on the specified configuration
    template. 
    
    It is currently built to run on any of those datasets. It can also 
    run routing and/or a snow routine if specified. 
    
    Required Parameters:
        Ptar: Precipitation for the basin, only used in Yalan Data runs
        Ttar: Temperature for the basin, only used in Yalan Data runs
        PETtar: PET for the basin, only used in Yalan Data runs
        AREAtar: Area of basin in KM^2
        DEPTHtar: Soil depth of basin 
        file_paramset: The path to the generated parameters for the CFE model
        outfile: The path where you want to store the output files
        config_template: The CFE configuration file template used to run the BMI
    Optional Parameters:
        runRouting: Flag set in LSE config to run the routing 
            (default = True)
        runSHM: Flag set in LSE config to run the snow routine 
            (default = True)
        endIndex: The last index in the forcing data the model looks at when running (helps with speed) 
            (default is ~5 years w/ spin-up)
        
    Returns:
        Pandas Series or Boolean (depends on the data used)
            If AORC or NLDAS => Returns the observed flow in mm/hr
            If Yalan Data => Returns False in order to use observed flow built into the model
    '''
    
    CFEparam_dict, routingparam_dict, snowparam_dict = load_CFE_param(file_paramset, AREAtar, DEPTHtar)
    
    # meant to help with araki baseline runs
    if runAraki:
        # grab Araki CFE config file to run through LSE
        gageID = CFEparam_dict["gageID"]
        CFEconfig_file = f"/home/m10870071/scratch/Araki_CAMELS/config_files/{gageID}_araki.json"
    else:
        CFEconfig_file = generate_config(CFEparam_dict, file_paramset, config_template) 
        
    ##### run CFE using BMI #####

    # initialize
    cfe_instance = bmi_cfe.BMI_CFE(CFEconfig_file)
    cfe_instance.initialize()
    
    aorc, nldas = False, False
    # QObs is built into this data so we don't need to worry about it
    if "aorc" in config_template.lower():
        aorc = True
        ds = xr.open_dataset(cfe_instance.forcing_file) # aorc forcings are .nc files
        df_forcing = ds.to_dataframe().reset_index()

        # set up index as date
        df_forcing['date'] = pd.to_datetime(df_forcing['date'])
        df_forcing = df_forcing.set_index('date')
        
        # generate PET using the data
        # df_forcing = penman_monteith_pet_hourly(df_forcing)
        
    elif "mixed" in config_template.lower():
        aorc = True
        ds = xr.open_dataset(cfe_instance.forcing_file) # aorc forcings are .nc files
        df_forcing = ds.to_dataframe().reset_index()
        
        # get pet from nldas data
        with open(cfe_instance.compare_results_file, 'r') as f:
            df_pet = pd.read_csv(f)
            df_pet['date'] = pd.to_datetime(df_pet['date'])
            
        # add pet to the forcings
        df_forcing = pd.merge(
            df_forcing,
            df_pet[['date', 'potential_evaporation']],
            on='date',
            how='inner'
        )
    # QObs is from a separate file so we can grab it here
    # NEED TO SETUP DIFFERENTLY
    elif "nldas" in config_template.lower():
        nldas = True
        with open(cfe_instance.forcing_file, 'r') as f:
            df_forcing = pd.read_csv(f)
            df_forcing['date'] = pd.to_datetime(df_forcing['date'])

        with open(cfe_instance.compare_results_file, 'r') as f:
            df_obs = pd.read_csv(f)
            df_obs['date'] = pd.to_datetime(df_obs['date'])

        # add obs to the forcings
        df_forcing = pd.merge(
            df_forcing,
            df_obs[['date', 'QObs(mm/h)', 'QObs_CAMELS(mm/h)']],
            on='date',
            how='inner'
        )

        df_forcing = df_forcing.rename(columns = {'date': 'time', 'QObs_CAMELS(mm/h)': 'qobs_mm_per_hour'})
    else:
        # run yalan data
        nldas = False
        aorc = False

    if os.path.isfile(outfile):
        print("CFE output file exists: ", outfile)
        print("Skip CFE Modeling")
        
        return df_forcing["qobs_mm_per_hour"]

    # initialize => update for each time step
    else:
        outputs = cfe_instance.get_output_var_names()
        output_lists = {output:[] for output in outputs}
        output_lists["runoff_depth_mm"] = []
        
        # setup snow routine variables
        snow_storage = 0.0
        DDF = snowparam_dict["DDF"]
        T_thresh = snowparam_dict["T_thresh"]
        delta_K = 24.0 # scale to hourly

        if aorc:
            for precip, temp, pet in zip(df_forcing['APCP_surface'][:endIndex], df_forcing['TMP_2maboveground'][:endIndex], df_forcing["potential_evaporation"][:endIndex]):
                # run the snow routine
                if runSHM:
                    melt, rain, snow_storage = shm_snow_step(precip, temp, snow_storage, T_thresh=T_thresh, DDF=DDF, delta_K=delta_K)
                    precip_m = (rain + melt) * 0.001 # "effective" precip, mm/hr => m/hr
                else:
                    precip_m = precip * 0.001
                pet_m = pet * 2.7778e-7 # mm/hr => m/s

                cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_m)
                cfe_instance.set_value('water_potential_evaporation_flux', pet_m)
                cfe_instance.update()

                for output in outputs:
                    if output == "land_surface_water__runoff_depth":
                        output_lists["runoff_depth_mm"].append(cfe_instance.get_value(output) * 1000.0) # convert to mm
                        output_lists[output].append(cfe_instance.get_value(output)) 
                    else:
                        output_lists[output].append(cfe_instance.get_value(output))
        elif nldas:
            for precip, temp, pet in zip(df_forcing['total_precipitation'][:endIndex], df_forcing['temperature'][:endIndex], df_forcing["potential_evaporation"][:endIndex]):
                # run the snow routine
                if runSHM:
                    melt, rain, snow_storage = shm_snow_step(precip, temp, snow_storage, T_thresh=T_thresh, DDF=DDF, delta_K=delta_K)
                    precip_m = (rain + melt) * 0.001 # effective precip, mm/hr => m/hr
                else:
                    precip_m = precip * 0.001 # mm/hr => m/hr
                pet_m = pet * 2.7778e-7 # mm/hr => m/s

                cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_m)
                cfe_instance.set_value('water_potential_evaporation_flux', pet_m)
                cfe_instance.update()

                for output in outputs:
                    if output == "land_surface_water__runoff_depth":
                        output_lists["runoff_depth_mm"].append(cfe_instance.get_value(output) * 1000.0) # convert to mm
                        output_lists[output].append(cfe_instance.get_value(output)) 
                    else:
                        output_lists[output].append(cfe_instance.get_value(output))
        else:
            n_steps = min(len(Ptar), len(Ttar), len(PETtar))
            
            for n in range(n_steps):
                precip = Ptar[n].item()
                temp = Ttar[n].item()
                pet = PETtar[n].item()
                
                # run the snow routine
                if runSHM:
                    melt, rain, snow_storage = shm_snow_step(precip, temp, snow_storage, T_thresh=T_thresh, DDF=DDF, delta_K=delta_K)
                    precip_m = (rain + melt) * 0.001 / 24.0 # "effective" precip, mm/day => m/hr
                else:
                    precip_m = precip * 0.001 / 24.0 # mm/day => m/hr
                pet_m = pet * 0.001 / 86400.0 # mm/day => m/s

                cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_m)
                cfe_instance.set_value('water_potential_evaporation_flux', pet_m)
                cfe_instance.update()
                
                for output in outputs:
                    if output == "land_surface_water__runoff_depth":
                        output_lists["runoff_depth_mm"].append(cfe_instance.get_value(output) * 1000.0) # convert to mm
                        output_lists[output].append(cfe_instance.get_value(output)) 
                    else:
                        output_lists[output].append(cfe_instance.get_value(output))
                
        # finalize
        cfe_instance.finalize(print_mass_balance=False)
        
        if runRouting:
            output_lists["runoff_depth_mm"] = route_flow(routingparam_dict, output_lists["runoff_depth_mm"])
        
        # convert to numpy array for speed
        # not sure if actually needed
        output_arrays = {key: np.array(value) for key, value in output_lists.items()} 
                
        print("Saving output file to ", outfile)        
        save_CFE_output(outfile, output_arrays)
        
        if aorc is False and nldas is False:
            return False

        qobs = df_forcing["qobs_mm_per_hour"] 
        # qobs_filled = qobs.interpolate(method='time').ffill().bfill()
        
        # we need to return the observed flow to compare with
        # the lse is designed to look at the Yalan Data flow
        return qobs  # _filled