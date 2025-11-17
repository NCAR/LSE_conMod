import pandas as pd
import numpy as np
import os, sys, glob, toml, multiprocessing

sys.path.append('../auxiliary/')
from parameter import generate_initial_parameter_sets
from evaluate import cal_metrics

sys.path.append('../HBV/')
# from HBV_KM import run_HBV_OneBasin_OneParamSet, load_HBV_output
from HBV import run_HBV_OneBasin_OneParamSet, load_HBV_output

sys.path.append('../CFE/')
from runCFE import run_CFE_OneBasin_OneParamSet, load_CFE_output


def run_modelflow_OneBasin_OneParamSet(model, tarparam, iter, gageID, tarbasin, Ptar, Ttar, PETtar, AREAtar, DEPTHtar, config_template, outpath_parent, outpath_basin, eval_indexStart, eval_indexEnd, eval_obs):
    
    # Target parameter set
    file_paramset = f'{outpath_parent}/{gageID}/param_sets/paramset_iter{iter}_trial{tarparam}.pkl'
    
    # Run selected model
    # HBV does not need a config file
    outfile = f'{outpath_basin}/hbv_iter{iter}_trial{tarparam}.pkl'
    if model == "HBV":
        run_HBV_OneBasin_OneParamSet(Ptar, Ttar, PETtar, file_paramset, outfile)
    elif model == "CFE":
        qobs = run_CFE_OneBasin_OneParamSet(Ptar, Ttar, PETtar, AREAtar, DEPTHtar, file_paramset, outfile, config_template, 52620)
        # only if using AORC or NLDAS
        if qobs is not False:
            eval_indexStart = 8760
            eval_indexEnd = 52608
            eval_obs = qobs[eval_indexStart:eval_indexEnd+1]
    else:
        raise ValueError("Invalid value provided")

    # Evaluate model output
    outfile_metric = f'{outpath_basin}/eval_iter{iter}_trial{tarparam}.csv'
    hbv_out = load_HBV_output(outfile)
    
    # get output depending on model
    if model == "HBV":
        eval_sim = hbv_out['flow_sim'][eval_indexStart:eval_indexEnd+1]
    elif model == "CFE":
        # eval_sim = hbv_out['flow_sim'][eval_indexStart:eval_indexEnd+1]
        eval_sim = hbv_out['runoff_depth_mm'][eval_indexStart:eval_indexEnd+1]
    
    df_tar_metric = cal_metrics(eval_obs, eval_sim, outfile_metric, overwrite=False)
    
    # Print output
    kge = df_tar_metric['kge2009'].values[0]
    nse = df_tar_metric['nse'].values[0]
    print(f"// Basin-{tarbasin}: {gageID}; ParamSet-{tarparam} // -> KGE={kge:.2f}; NSE={nse:0.2f}")


def merge_parameter_metric_OneBasin_OneIter(path_archive, iterflag, num_trial):
    # merge parameters and metric values from all runs to one csv file

    outfile_metric = f'{path_archive}/outputs/iter{iterflag}_all_metric.csv'
    outfile_param = f'{path_archive}/param_sets/iter{iterflag}_all_param.csv'
    if (not os.path.isfile(outfile_metric)) or (not os.path.isfile(outfile_param)):

        flag = True
        metric_values = []
        for i in range(num_trial):
            file_metric = f'{path_archive}/outputs/eval_iter{iterflag}_trial{i}.csv'

            if not os.path.isfile(file_metric):
                print(f'Warning! File does not exist: {file_metric}')
                continue
            
            df_met = pd.read_csv(file_metric)
            if flag:
                metric_names = list(df_met.columns)
                metric_values = np.nan * np.zeros([num_trial, len(metric_names)])
                flag = False

            metric_values[i, :] = df_met.iloc[0].values

        flag = True
        param_meanvalues = []
        for i in range(num_trial):
            file_param = f'{path_archive}/param_sets/paramset_iter{iterflag}_trial{i}.pkl'

            if not os.path.isfile(file_param):
                print(f'Warning! File does not exist: {file_param}')
                continue
            
            df_param = pd.read_pickle(file_param)
            if flag:
                param_names = list(df_param['Parameter'].values)
                param_meanvalues = np.nan * np.zeros([num_trial, len(param_names)])
                flag = False
    
            param_meanvalues[i, :] = np.array([np.nanmean(v) for v in df_param['Value'].values])

            
        # print(f'Write all metrics for {num_trial} trials in iteration {iterflag} to {outfile_metric}')
        dfout = pd.DataFrame(metric_values, columns=metric_names)
        dfout.to_csv(outfile_metric, index=False)
    
        # print(f'Write all parameters (mean value) for {num_trial} trials in iteration {iterflag} to {outfile_param}')
        dfout = pd.DataFrame(param_meanvalues, columns=param_names)
        dfout.to_csv(outfile_param, index=False)

    return outfile_metric, outfile_param


def extract_best_paramset_OneBasin_AllIter(outfile_optmz_paramset, path_archive, iterend, objfunc='nse', method='max', external_iter0_path=''):
    # extract the best parameter set based on all existing iterations 

    if os.path.isfile(outfile_optmz_paramset):
        return

    # find best iter/trial number
    df_allmet = []
    for iter in range(iterend+1):
        if iter==0 and len(external_iter0_path)>0:
            file_metric = f'{external_iter0_path}/outputs/iter{iter}_all_metric.csv'
        else:
            file_metric = f'{path_archive}/outputs/iter{iter}_all_metric.csv'
        dfi = pd.read_csv(file_metric)
        dfi['iter'] = iter
        dfi['trial'] = np.arange(len(dfi))
        df_allmet.append(dfi)
    df_allmet = pd.concat(df_allmet)

    if method == 'max':
        index = np.nanargmax(df_allmet[objfunc].values)
    elif method == 'min':
        index = np.nanargmin(df_allmet[objfunc].values)
        
    best_iter = df_allmet.iloc[index]['iter'].astype(np.int32)
    best_trial = df_allmet.iloc[index]['trial'].astype(np.int32)

    # copy best parameter set
    if best_iter == 0 and len(external_iter0_path)>0:
        file_param_source = f'{external_iter0_path}/param_sets/paramset_iter{best_iter}_trial{best_trial}.pkl'
    else:
        file_param_source = f'{path_archive}/param_sets/paramset_iter{best_iter}_trial{best_trial}.pkl'
        
    os.system(f'cp {file_param_source} {outfile_optmz_paramset}')

    source_info_file = outfile_optmz_paramset + '.sourceinfo'
    with open(source_info_file, 'w') as f:
        f.write(f'iter: {best_iter}\n')
        f.write(f'iter: {best_trial}\n')
    
    
def run_model_on_best_paramset(outfile_optmz_paramset, outfile_optmz_simu, outfile_metric, path_archive, iterend, P, T, PET, flowobs, eval_indexStart, eval_indexEnd, external_iter0_path=''):

    extract_best_paramset_OneBasin_AllIter(outfile_optmz_paramset, path_archive, iterend, objfunc='nse', method='max', external_iter0_path=external_iter0_path)

    run_HBV_OneBasin_OneParamSet(P, T, PET, outfile_optmz_paramset, outfile_optmz_simu)

    # Evaluate model output
    hbv_out = load_HBV_output(outfile_optmz_simu)
    eval_sim = hbv_out['flow_sim'][eval_indexStart:eval_indexEnd+1]
    df_tar_metric = cal_metrics(flowobs[eval_indexStart:eval_indexEnd+1], eval_sim, outfile_metric, overwrite=False)
    
    # Print output
    kge = df_tar_metric['kge2009'].values[0]
    nse = df_tar_metric['nse'].values[0]
    print(f"evaluation: KGE={kge:.2f}; NSE={nse:0.2f}")

