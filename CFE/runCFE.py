# Helper file to couple CFE to LSE similar to how the HBV is running 
# Similar structure to ../HBV/HBV.py
# Created by Rhys Pulling

import pickle, os, sys, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import bmi_cfe

# NEED TO SET UP TO GRAB PARAMS ACCURATELY
# Grab config file?
def load_CFE_param(infile_param):
    # read CFE parameters from a pickle file

    try:
        df = pd.read_pickle(infile_param)
    except:
        df = pd.read_csv(infile_param)
    
    param1 = ['soil_params.D', 'soil_params.bb', 'soil_params.satdk', 'soil_params.satpsi', 'soil_params.slop', 'soil_params.smcmax', 'soil_params.wltsmc', 'soil_params.expon', 
              'soil_params.expon_secondary', 'max_gw_storage', 'Cgw', 'expon', 'alpha_fc', 'K_nash', 'K_lf', 'nash_storage', 'a_Xinanjiang_inflection_point', 'b_Xinanjiang_shape_param', 
              'x_Xinanjiang_shape_param']
    # param2 = ['rout_a', 'rout_b'] # might need to include later

    param1_dict = {}
    for p in param1:
        param1_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    '''
    param2_dict = {}
    for p in param2:
        param2_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]
    '''

    return param1_dict #, param2_dict

def save_CFE_output(filename, data, overwrite=True):
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

def load_CFE_output(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# helper function for generating config
def replace_placeholders(obj, param_dict):
    if isinstance(obj, dict):
        return {k: replace_placeholders(v, param_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_placeholders(item, param_dict) for item in obj]
    elif isinstance(obj, str):
        # Match a placeholder like {{key}}
        match = re.fullmatch(r"{{(.*?)}}", obj)
        if match:
            key = match.group(1)
            return param_dict.get(key, obj) 
        else:
            return obj
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

def run_CFE_OneBasin_OneParamSet(Ptar, Ttar, PETtar, file_paramset, outfile, config_template):
    '''
    Use the Basic Model Interface to interact with the CFE to 
    run it with the given parameters. 
    '''

    if os.path.isfile(outfile):
        print("CFE output file exists: ", outfile)
        print("Skip CFE Modeling")

    # initialize => update for each time step
    else:
        CFEparam_dict = load_CFE_param(file_paramset)
        CFEconfig_file = generate_config(CFEparam_dict, file_paramset, config_template) 
        
        ##### run CFE using BMI #####
        
        # initialize
        cfe_instance = bmi_cfe.BMI_CFE(CFEconfig_file)
        cfe_instance.initialize()
        
        outputs = cfe_instance.get_output_var_names()
        output_lists = {output:[] for output in outputs}

        n_steps = len(Ptar)

        for n in range(n_steps):
            precip = Ptar[n].item()
            temp = Ttar[n].item()
            pet = PETtar[n].item()

            cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip)
            cfe_instance.update()
            for output in outputs:
                output_lists[output].append(cfe_instance.get_value(output))
            
        # finalize
        cfe_instance.finalize(print_mass_balance=False)
                
        print("Saving output file to ", outfile)        
        save_CFE_output(outfile, output_lists)

