import pandas as pd
import numpy as np
import os, sys, glob, toml, multiprocessing

sys.path.append('../auxiliary/')
from parameter import generate_initial_parameter_sets
from evaluate import cal_metrics

sys.path.append('../HBV/')
from HBV import run_HBV_OneBasin_OneParamSet, load_HBV_output


def run_modelflow_OneBasin_OneParamSet(tarparam, iter, gageID, tarbasin, Ptar, Ttar, PETtar, outpath_parent, outpath_basin, eval_indexStart, eval_indexEnd, eval_obs):
    
    # Target parameter set
    file_paramset = f'{outpath_parent}/{gageID}/param_sets/paramset_iter{iter}_trial{tarparam}.pkl'

    # Run HBV
    outfile_hbv = f'{outpath_basin}/hbv_iter{iter}_trial{tarparam}.pkl'
    run_HBV_OneBasin_OneParamSet(Ptar, Ttar, PETtar, file_paramset, outfile_hbv)

    # Evaluate model output
    outfile_metric = f'{outpath_basin}/eval_iter{iter}_trial{tarparam}.csv'
    hbv_out = load_HBV_output(outfile_hbv)
    eval_sim = hbv_out['flow_sim'][eval_indexStart:eval_indexEnd+1]
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
