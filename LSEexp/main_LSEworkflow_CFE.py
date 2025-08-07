# Run LSE-based HBV calibration
# Consistent with NBdemo codes but with some structure change

import glob, os, sys, time, toml, multiprocessing
from datetime import datetime
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append("../auxiliary/")
from data_processing import *
from parameter import *
from run_model import *
from evaluate import cal_metrics

sys.path.append("../HBV/")
from HBV import load_HBV_output, run_HBV_OneBasin_OneParamSet

sys.path.append("../CFE/")
from runCFE import load_CFE_output, run_CFE_OneBasin_OneParamSet


##########################################################################################
# Workflow control

config_file = sys.argv[1]
iter_now = int(sys.argv[2])

# if ncpu>1, parallelization will be used
if len(sys.argv) == 4:
    ncpu = int(sys.argv[3])
else:
    ncpu = 1

run_CFE = True # if False, HBV won't be run. this can be used when you know model simulations have been generated
run_LSE = True # if False, only run initial sampling + model simulation (stop before LSE)

print('Configuration file:',config_file)
print('Processing iteration', iter_now)
print('CPU number:', ncpu)

config = toml.load(config_file)


##########################################################################################
# initial sampling. Only used for iteration-0

# whether use iter-0 results from another folder to be consistent
external_iter0 = False # if True, the code will use iter-0 (random sampling) from another folder
external_iter0_path = ""
if 'external_iter0' in config['output']:
    external_iter0 = config['output']['external_iter0']
    if external_iter0 == True:
        external_iter0_path = config['output']['external_iter0_path']

if iter_now == 0 and external_iter0 == False:
    print('Generate random samples for iteration-0')

    # Configurations and data
    sampling_method = config['emulator']['sampling_method']
    num_init = config['emulator']['num_init']
    outpath_parent = config['output']['outpath_parent']
    
    file_parameter_list = config['data']['file_parameter_list']
    inpath_data = config['data']['inpath_data']
    
    print('sampling_method:', sampling_method)
    print('num_init:', num_init)
    print('outpath_parent:', outpath_parent)
    
    selected_camels = load_Yalan_selected_camels_id(inpath_data)
    print(len(selected_camels))
    
    # Generate initial samples
    df_param = pd.read_csv(file_parameter_list)
    # display(df_param)
    
    for i in range(len(selected_camels)):
        idi = selected_camels[i]
        print('processing', idi)
    
        outpathi = f'{outpath_parent}/{idi}/param_sets'
        os.makedirs(outpathi, exist_ok=True)
    
        init_param_filelist = generate_initial_parameter_sets(file_parameter_list, sampling_method, outpathi, num_init, adddefault=False)


##########################################################################################
# Running Model

################### Load data

# load configuration

# config_file = "config_LSEexp1.toml"
# config = toml.load(config_file)

model = config['model']['model_to_run']
config_template = config['model']['config_temp']

inpath_data = config['data']['inpath_data']
file_parameter_list = config['data']['file_parameter_list']

outpath_parent = config["output"]["outpath_parent"]

train_run_period = config["dates"]["train_run_period"]
train_eval_period = config["dates"]["train_eval_period"]
eval_indexStart, eval_indexEnd = get_index_of_nested_period(train_run_period, train_eval_period)

sampling_method = config["emulator"]["sampling_method"]
num_init = config["emulator"]["num_init"]
objfunc = config["emulator"]["objfunc"]
num_per_iter = config["emulator"]["num_per_iter"]
if iter_now == 0:
    num_trial = config["emulator"]["num_init"]
else:
    num_trial = num_per_iter

# display(config)


# forcing, flow, and basin data
startDate = train_run_period[0]
endDate = train_run_period[1]
print("run model period:", startDate, "-", endDate)

forcing_train, target_train, attr_train, date_train, forcingLst, allbasin_IDs, attrLst = load_Yalan_data(inpath_data, startDate, endDate)
print("forcing shape", forcing_train.shape)

num_basin = len(allbasin_IDs)
print("basin number", num_basin)

# parameter list containing upper / lower bounds
df_param_bound = pd.read_csv(file_parameter_list)
df_param_bound.head()


################### Run the real model
# run_modelflow_OneBasin_OneParamSet is a high-level function wrapping up many components

print("#### Running model simulations ####")

if run_CFE == False:
    print('Do not run CFE because run_CFE is False')

elif iter_now == 0 and external_iter0 == True:
    print('external_iter0 is True for iter-0. Find model outputs in ', external_iter0_path)

else:

    t01 = time.time()
    
    for tarbasin in range(num_basin):
        t11 = time.time()

        gageID = allbasin_IDs[tarbasin]
        outpath_basin = f"{outpath_parent}/{gageID}/outputs/"
        os.makedirs(outpath_basin, exist_ok=True)

        Ptar = forcing_train[tarbasin, :, forcingLst.index("prcp")][:, np.newaxis]
        Ttar = forcing_train[tarbasin, :, forcingLst.index("tmean")][:, np.newaxis]
        PETtar = forcing_train[tarbasin, :, forcingLst.index("pet")][:, np.newaxis]

        eval_obs = target_train[tarbasin, :, 0][eval_indexStart : eval_indexEnd + 1]

        if ncpu == 1:
            # serial
            for tarparam in range(num_trial):
                run_modelflow_OneBasin_OneParamSet(model, tarparam, iter_now, gageID, tarbasin, Ptar, Ttar, PETtar, config_template,
                                                   outpath_parent, outpath_basin, eval_indexStart, eval_indexEnd, eval_obs)
        else:
            # parallel
            alltrial = [(model, tarparam, iter_now, gageID, tarbasin, Ptar, Ttar, PETtar, config_template, outpath_parent,
                         outpath_basin, eval_indexStart, eval_indexEnd, eval_obs) for tarparam in range(num_trial)]
            
            with multiprocessing.Pool(processes=ncpu) as pool:
                pool.starmap(run_modelflow_OneBasin_OneParamSet, alltrial)

        t12 = time.time()
        # print(f"HBV time cost for basin-{tarbasin} gageID (s):", t12 - t11)

    t02 = time.time()
    print(f"Model time cost for all basins (s):", t02 - t01)


################### File post-processing
# merge parameter / evaluation files for one iteration to one file to faciliate later analysis

if iter_now == 0 and external_iter0 == True:
    print('external_iter0 is True for iter-0. Find param/eval merged outputs in ', external_iter0_path)

else:
    
    if ncpu == 1:
        # Serial execution with progress bar
        for tarbasin in tqdm(range(num_basin), desc="Merging param/eval for all basins"):
            gageID = allbasin_IDs[tarbasin]
            path_archive = f"{outpath_parent}/{gageID}/"
            merge_parameter_metric_OneBasin_OneIter(path_archive, iter_now, num_trial)
    
    else:
        args = [(f"{outpath_parent}/{allbasin_IDs[tarbasin]}/", iter_now, num_trial) for tarbasin in range(num_basin) ]
        # Use multiprocessing with a progress bar
        with multiprocessing.Pool(processes=ncpu) as pool:
            pool.starmap(merge_parameter_metric_OneBasin_OneIter, args) 

# stop program if flag is set to False
print("run_LSE is set to ", run_LSE)
if run_LSE == False:
    print("Exiting before running LSE.")
    sys.exit()            

##########################################################################################
# Large-sample emulator: Train and Optimization

################### Data preparation
outpath_lse = f'{outpath_parent}/LargeSampleEmulator'
os.makedirs(outpath_lse, exist_ok=True)
print('LSE outpath:', outpath_lse)

###### generate files containing parameter, metric, and basinid information
iterend = iter_now+1

if external_iter0 == True:
    external_iter0_info = {}
    external_iter0_info['external_iter0_path'] = external_iter0_path
    external_iter0_info['external_iter0_pathlse'] = f'{external_iter0_path}/LargeSampleEmulator'
else:
    external_iter0_info = {}

df_param, df_metric, df_basinid = load_allbasin_params_metrics_AllIter(outpath_parent, outpath_lse, allbasin_IDs, iterend, external_iter0_info)

###### prepare attribute

# attribute list used in CTSM/SUMMA original studies
# attrLst_train = ['elev_mean', 'slope_mean', 'area_gages2', 'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 
#                    'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'forest_frac', 'lai_max', 'lai_diff', 
#                    'dom_land_cover', 'dom_land_cover_frac', 'soil_depth_pelletier', 'soil_depth_statgso', 'soil_porosity', 
#                    'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'carb_rocks_frac', 'geol_permeability'] # from Xie2021
attrLst_train = config['emulator']['attrlist']
if attrLst_train == 'All':
    attrLst_train = attrLst

# remove attributes not included in target train attributes
flag = np.ones(len(attrLst))
for i in range(len(attrLst)):
    if not attrLst[i] in attrLst_train:
        flag[i] = 0
    if np.sum(np.isnan(attr_train[:,i]))>0:
        print('attribute has nan value. exclude ', attrLst[i])
        flag[i] = 0    

attrLst_filtered = list(np.array(attrLst)[flag==1])
print('Attribute - Target number:', len(attrLst_train))
print('Attribute - YSong number:', len(attrLst))
print('Attribute - YSong_filtered number:', len(attrLst_filtered))
for a in attrLst_train:
    if not a in attrLst_filtered:
        print(f'Missing {a} in YSong_filtered compared to target attributes')

# make a dataframe
df_att = pd.DataFrame(attr_train[:,flag==1], columns=attrLst_filtered)
df_att['hru_id'] = allbasin_IDs


###### Prepare model input
df_input = df_param.copy()
df_input["hru_id"] = df_basinid["hru_id"]
df_input = df_input.merge(df_att[attrLst_filtered + ["hru_id"]], on="hru_id", how="left")
df_input = df_input.drop(["hru_id"], axis=1)

inputnames = list(df_param.columns) + attrLst_filtered
print(f'Input features number: {len(inputnames)} \n', inputnames)

x_all = df_input[inputnames].values.copy()
print("Input shape:", x_all.shape)

###### Prepare model training target
y_all = get_objective_function(df_metric, objfunc)
print('Output shape:', y_all.shape)

################### Train an emulator

t1 = time.time()

# Train a random forest emulator
iterend = iter_now + 1

outfile_lse = f'{outpath_lse}/RF_emulator_for_iter{iterend}.pkl'
if os.path.isfile(outfile_lse):
    with open(outfile_lse, 'rb') as file:
        em_model = pickle.load(file)
        
else:
    modelconfig = {'n_estimators': 100, 'random_state': 42, 'max_depth': 40}
    em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpu)
    em_model.fit(x_all, y_all)
    with open(outfile_lse, 'wb') as file:
        pickle.dump(em_model, file)

# print em_model training accuracy
y_all_pred = em_model.predict(x_all)
print('rmse of em:', np.mean( (y_all_pred - y_all) ** 2 ) ** 0.5)

t2 = time.time()
print('LSE train time cost (s):', t2-t1)

################### Generate optimized parameter values for each basin
# check parameter match
if np.any( df_param.columns != df_param_bound['Parameter'] ):
    sys.exit('df_param and df_param_bound parameters do not match')


# optimize for each basin
times = config['emulator']['optmz_times'] # times*num_per_iter is the actual number of optimization algorithm running times
num_objfunc = 1

iterend = iter_now+ 1
modelparam_lb = df_param_bound['Lower'].values
modelparam_ub = df_param_bound['Upper'].values
param_calibflag = np.zeros( len(df_input.columns) )
param_calibflag[:len(modelparam_lb)]=1
param_names = df_input.columns.values

if ncpu > 1:
    args = [
        (
            f'{outpath_parent}/{allbasin_IDs[i]}/param_sets',  # outpath_param
            iterend, 
            em_model, 
            np.hstack([modelparam_lb, df_att[attrLst_filtered].iloc[i].values]),  # param_lb
            np.hstack([modelparam_ub, df_att[attrLst_filtered].iloc[i].values]),  # param_ub
            param_calibflag, 
            param_names, 
            num_per_iter, 
            num_objfunc,
            times
        )
        for i in range(num_basin)
    ]

    with multiprocessing.Pool(processes=ncpu) as pool:
        pool.starmap(generate_optimized_paramsets, args)
        
else:

    for i in range(len(allbasin_IDs)):
        outpath_param = f'{outpath_parent}/{allbasin_IDs[i]}/param_sets'
        os.makedirs(outpath_param, exist_ok=True)
        
        basin_attr = df_att[attrLst_filtered].iloc[i].values
        param_lb = np.hstack([modelparam_lb, basin_attr])
        param_ub = np.hstack([modelparam_ub, basin_attr])
        
        generate_optimized_paramsets(outpath_param, iterend, em_model, param_lb, param_ub, param_calibflag, param_names, num_per_iter, num_objfunc=num_objfunc, times=times)

##########################################################################################
