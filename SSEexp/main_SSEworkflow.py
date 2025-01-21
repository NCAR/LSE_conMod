# Run SSE-based HBV calibration

import glob, os, sys, time, toml, multiprocessing
from datetime import datetime
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from SSE_optmz import surrogate_model_train_and_optmz

sys.path.append("../auxiliary/")
from data_processing import *
from parameter import *
from run_model import *
from evaluate import cal_metrics

sys.path.append("../HBV/")
from HBV_KM import load_HBV_output, run_HBV_OneBasin_OneParamSet


##########################################################################################
# Workflow control

config_file = sys.argv[1]
iter_now = int(sys.argv[2])

# if ncpu>1, parallelization will be used
if len(sys.argv) == 4:
    ncpu = int(sys.argv[3])
else:
    ncpu = 1

run_HBV = True # if False, HBV won't be run. this can be used when you know model simulations have been generated

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

if 'subset_flag' in config['data'] and config['data']['subset_flag'] == True:
    file_subsetID = config['data']['file_subsetID']
else:
    file_subsetID = ''

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
    
    selected_camels = load_Yalan_selected_camels_id(inpath_data, file_subsetID)
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
# Running HBV

################### Load data

# load configuration

# config_file = "config_LSEexp1.toml"
# config = toml.load(config_file)

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

forcing_train, target_train, attr_train, date_train, forcingLst, allbasin_IDs, attrLst = load_Yalan_data(inpath_data, startDate, endDate, file_subsetID)
print("forcing shape", forcing_train.shape)

num_basin = len(allbasin_IDs)
print("basin number", num_basin)

# parameter list containing upper / lower bounds
df_param_bound = pd.read_csv(file_parameter_list)
df_param_bound.head()


################### Run the real model
# run_modelflow_OneBasin_OneParamSet is a high-level function wrapping up many components

if run_HBV == False:
    print('Do not run HBV because run_HBV is False')

elif iter_now == 0 and external_iter0 == True:
    print('external_iter0 is True for iter-0. Find model outputs in ', external_iter0_path)

else:

    t01 = time.time()
    
    for tarbasin in range(num_basin):
        t11 = time.time()

        gageID = allbasin_IDs[tarbasin]
        outpath_basin = f"{outpath_parent}/{gageID}/outputs/"
        os.makedirs(outpath_basin, exist_ok=True)

        Ptar = forcing_train[tarbasin, :, forcingLst.index("prcp")]
        Ttar = forcing_train[tarbasin, :, forcingLst.index("tmean")]
        PETtar = forcing_train[tarbasin, :, forcingLst.index("pet")]

        eval_obs = target_train[tarbasin, :, 0][eval_indexStart : eval_indexEnd + 1]

        if ncpu == 1:
            # serial
            for tarparam in range(num_trial):
                run_modelflow_OneBasin_OneParamSet(tarparam, iter_now, gageID, tarbasin, Ptar, Ttar, PETtar,
                                                   outpath_parent, outpath_basin, eval_indexStart, eval_indexEnd, eval_obs)
        else:
            # parallel
            alltrial = [(tarparam, iter_now, gageID, tarbasin, Ptar, Ttar, PETtar, outpath_parent, outpath_basin, 
                         eval_indexStart, eval_indexEnd, eval_obs) for tarparam in range(num_trial)]
            
            with multiprocessing.Pool(processes=ncpu) as pool:
                pool.starmap(run_modelflow_OneBasin_OneParamSet, alltrial)

        t12 = time.time()
        # print(f"HBV time cost for basin-{tarbasin} gageID (s):", t12 - t11)

    t02 = time.time()
    print(f"HBV time cost for all basins (s):", t02 - t01)


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



##########################################################################################
# Single-site emulator for each basin: Train and Optimization


if external_iter0 == True:
    external_iter0_info = {}
    external_iter0_info['external_iter0_path'] = external_iter0_path
    external_iter0_info['external_iter0_pathlse'] = f'{external_iter0_path}/LargeSampleEmulator'
else:
    external_iter0_info = {}


modelparam_lb = df_param_bound['Lower'].values
modelparam_ub = df_param_bound['Upper'].values
param_names = df_param_bound['Parameter'].values


iterend = iter_now+1

if ncpu == 1:

    for gageID in allbasin_IDs:
    
        df_param, df_metric = load_oneasin_params_metrics_AllIter(outpath_parent, gageID, iterend, external_iter0_info)
    
        x_all = df_param.values.copy()
        print("Input shape:", x_all.shape)
    
        y_all = get_objective_function(df_metric, objfunc)
        y_all = y_all[:, np.newaxis]
        print('Output shape:', y_all.shape)
    
        outpath_i = f'{outpath_parent}/{gageID}/param_sets'
        surrogate_model_train_and_optmz(x_all, y_all, modelparam_lb, modelparam_ub, param_names, outpath_i, iter_now, num_per_iter)
    
        break

else:
    # Prepare the arguments for each surrogate_model_train_and_optmz call
    args_list = []
    for gageID in allbasin_IDs:
        # Load data for the current basin
        df_param, df_metric = load_oneasin_params_metrics_AllIter(outpath_parent, gageID, iterend, external_iter0_info)
    
        x_all = df_param.values.copy()
        print("Input shape:", x_all.shape)
    
        y_all = get_objective_function(df_metric, objfunc)
        y_all = y_all[:, np.newaxis]
        print('Output shape:', y_all.shape)
    
        outpath_i = f'{outpath_parent}/{gageID}/param_sets'
    
        # Add arguments for this basin to the list
        args_list.append((x_all, y_all, modelparam_lb, modelparam_ub, param_names, outpath_i, iter_now, num_per_iter))
    
    # Function to unpack arguments and call surrogate_model_train_and_optmz
    def parallel_worker(args):
        surrogate_model_train_and_optmz(*args)
    
    # Use multiprocessing to parallelize the calls
    if __name__ == "__main__":
        with multiprocessing.Pool() as pool:
            pool.map(parallel_worker, args_list)


##########################################################################################
# (optional) evaluate the best parameter after each iteration is done


if external_iter0 == True:
    external_iter0_path = config['output']['external_iter0_path']
else:
    external_iter0_path = ''
    
if ('eval_optmz_param' in config['output']) and (config['output']['eval_optmz_param']==True) and (iter_now>0):

    # evaluate optimized parameter sets (training period)
    eval_indexStart, eval_indexEnd = get_index_of_nested_period(train_run_period, train_eval_period)
    tasks = []
    for tarbasin in range(len(allbasin_IDs)):
        gageID = allbasin_IDs[tarbasin]
        P = forcing_train[tarbasin, :, forcingLst.index('prcp')]
        T = forcing_train[tarbasin, :, forcingLst.index('tmean')]
        PET = forcing_train[tarbasin, :, forcingLst.index('pet')]
        flowobs = target_train[tarbasin, :, 0]
    
        path_archive = f'{outpath_parent}/{gageID}'
        os.makedirs(f'{outpath_parent}/{gageID}/optmz_simu', exist_ok=True)
        outfile_optmz_paramset = f'{outpath_parent}/{gageID}/optmz_simu/best_param_iter{iter_now}.csv'
        outfile_optmz_simu = f'{outpath_parent}/{gageID}/optmz_simu/bestparam_train_simu_iter{iter_now}.pkl'
        outfile_metric = f'{outpath_parent}/{gageID}/optmz_simu/bestparam_train_eval_iter{iter_now}.csv'

        if len(external_iter0_path)>0:
            extpath = f'{external_iter0_path}/{gageID}'
        else:
            extpath = ''
        
        tasks.append((outfile_optmz_paramset, outfile_optmz_simu, outfile_metric, path_archive, iter_now, P, T, PET, flowobs, eval_indexStart, eval_indexEnd, extpath))

    if ncpu == 1:
        # Serial processing
        for task in tasks:
            run_model_on_best_paramset(*task)
    else:
        # Parallel processing
        with Pool(ncpu) as pool:
            pool.starmap(run_model_on_best_paramset, tasks)
    
    
    # evaluate optimized parameter sets (testing period)
    test_run_period = config["dates"]["test_run_period"]
    test_eval_period = config["dates"]["test_eval_period"]
    startDate = test_run_period[0]
    endDate = test_run_period[1]
    print("run model period:", startDate, "-", endDate)
    forcing_test, target_test, attr_test, date_test, forcingLst, allbasin_IDs, attrLst = load_Yalan_data(inpath_data, startDate, endDate, file_subsetID)
    
    eval_indexStart, eval_indexEnd = get_index_of_nested_period(test_run_period, test_eval_period)
    
    tasks = []
    for tarbasin in range(len(allbasin_IDs)):
        gageID = allbasin_IDs[tarbasin]
        P = forcing_test[tarbasin, :, forcingLst.index('prcp')]
        T = forcing_test[tarbasin, :, forcingLst.index('tmean')]
        PET = forcing_test[tarbasin, :, forcingLst.index('pet')]
        flowobs = target_test[tarbasin, :, 0]

        path_archive = f'{outpath_parent}/{gageID}'
        os.makedirs(f'{outpath_parent}/{gageID}/optmz_simu', exist_ok=True)
        outfile_optmz_paramset = f'{outpath_parent}/{gageID}/optmz_simu/best_param_iter{iter_now}.csv'
        outfile_optmz_simu = f'{outpath_parent}/{gageID}/optmz_simu/bestparam_test_simu_iter{iter_now}.pkl'
        outfile_metric = f'{outpath_parent}/{gageID}/optmz_simu/bestparam_test_eval_iter{iter_now}.csv'

        if len(external_iter0_path)>0:
            extpath = f'{external_iter0_path}/{gageID}'
        else:
            extpath = ''
        
        tasks.append((outfile_optmz_paramset, outfile_optmz_simu, outfile_metric, path_archive, iter_now, P, T, PET, flowobs, eval_indexStart, eval_indexEnd, extpath))
    
    if ncpu == 1:
        # Serial processing
        for task in tasks:
            run_model_on_best_paramset(*task)
    else:
        # Parallel processing
        with Pool(ncpu) as pool:
            pool.starmap(run_model_on_best_paramset, tasks)
            
##########################################################################################

