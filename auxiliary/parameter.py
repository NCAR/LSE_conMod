# created by Guoqiang Tang in 2024-12

import os, sys, subprocess, pickle, random
import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

# sys.path.append('/glade/u/home/guoqiang/CTSM_repos/ctsm_optz/MO-ASMO/src/')
sys.path.append('/beegfs/sets/aw-ciroh/projects/tmpdir/src')
import sampling, gp, NSGA2



def generate_initial_parameter_sets(file_parameter_list, sampling_method, outpath, num_init=-1, adddefault=True):
    # example parameters
    # sampling_method = 'lh'  # lh: LatinHypercubeDesign, slh: SymmetricLatinHypercubeDesign, glp: GoodLatticePointsDesign
    # param_upper_bound = {'param1': np.array(15), 'param2': np.array([1, 2, 3])}
    # param_lower_bound = {'param1': np.array(3), 'param2': np.array([0.2, 1.5, 2.2])}
    # if adddefault=True, the first parameter set will be replaced by the default parameter of CTSM

    os.makedirs(outpath, exist_ok=True)

    df_calibparam = pd.read_csv(file_parameter_list)
    param_upper_bound = df_calibparam['Upper'].values
    param_lower_bound = df_calibparam['Lower'].values

    # dimension sizes
    num_param = len(param_lower_bound) # number of parameters to be calibrated
    if not num_init > 0:
        num_init = num_param * 20 # number of initial samples (i.e., initial model runs). A proper initial sample size should be 15–20 times the number of parameters (Gong et al., 2015)
        
    # check whether parameter files have been generated
    flag = False
    outfiles_all = []
    for i in range(num_init):
        outfile = f'{outpath}/paramset_iter0_trial{i}.pkl'
        outfiles_all.append(outfiles_all)
        if not os.path.isfile(outfile):
            flag = True
            break
    
    if flag == False:
        print('All ini parameter csv/pkl files have been generated. Skip this step')
    else:
        # get initial factors between 0 and 1 which will be used to scale real parameters
        # init_factors: [num_init, num_param]
        if sampling_method == 'lh':
            init_factors = sampling.lh(num_init, num_param)
        elif sampling_method == 'slh':
            init_factors = sampling.slh(num_init, num_param)
        elif sampling_method == 'glp': # glp is used by Gong et al., 2015
            init_factors = sampling.glp(num_init, num_param)
        else:
            sys.exit('Unknown sampling method!')

        # save factors
        df_factor = pd.DataFrame(init_factors, columns=df_calibparam['Parameter'].values)
        df_factor.to_csv(f'{outpath}/paramset_iter0_scalefactors.csv', index=False)
        
        # load default parameter dataframe (file will be saved after first generation)
        param0 = df_calibparam['Default'].values

        # generate a dataframe for every set of parameters and deal with binding parameters
        outfiles_all = []
        for i in range(num_init):
            #outfile = f'{outpath}/paramset_iter0_trial{i}.csv'
            outfile = f'{outpath}/paramset_iter0_trial{i}.pkl'
            # print('Generating parameter file:', outfile)
            dfi = df_calibparam.copy()

            if i == 0 and adddefault == True:
                print('For iteration 0, the default parameters will be used. scaling factors are not adopted')

                param_names = df_calibparam['Parameter'].values
                param_sources = df_calibparam['Source'].values
                
                factor0 = []
                for j in range(len(param_names)):
                    factor0.append( ( np.nanmean(param0[j]) - param_lower_bound[j] )/(param_upper_bound[j] - param_lower_bound[j]) )

                dfi['Value'] = param0
                dfi['Factor'] = factor0

            else:

                dfi['Factor'] = init_factors[i, :]
                meanparam = init_factors[i, :] * (param_upper_bound - param_lower_bound) + param_lower_bound
                dfi['Value'] = meanparam

            #dfi.to_csv(outfile, index=False)
            dfi.to_pickle(outfile)
            
            outfiles_all.append(outfiles_all)
    
    
    return outfiles_all



##################
# LSE calibration


def load_allbasin_params_metrics_OneIter(inpath_basins, outpath_lse, gageIDs, tariter):

    file_all_param = f'{outpath_lse}/camels{len(gageIDs)}_param_iter{tariter}.gz'
    file_all_metric = f'{outpath_lse}/camels{len(gageIDs)}_metric_iter{tariter}.gz'
    file_all_basinid = f'{outpath_lse}/camels{len(gageIDs)}_basinid_iter{tariter}.gz'
    
    if os.path.isfile(file_all_param) and os.path.isfile(file_all_metric) and os.path.isfile(file_all_basinid):
        df_param = pd.read_csv(file_all_param, compression='gzip')
        df_metric = pd.read_csv(file_all_metric, compression='gzip')
        df_basinid = pd.read_csv(file_all_basinid, compression='gzip')
    else:
    
        df_metric = pd.DataFrame()
        df_param = pd.DataFrame()
        flag = 0
    
        for i in range(len(gageIDs)):
            IDi = gageIDs[i]
            
            # load param
            file_param = f"{inpath_basins}/{IDi}/param_sets/iter{tariter}_all_param.csv"
            df1 = pd.read_csv(file_param)
            if df_param.empty:
                df_param = df1
            else:
                df_param = pd.concat([df_param, df1])
    
            # load metric
            file_metric = f"{inpath_basins}/{IDi}/outputs/iter{tariter}_all_metric.csv"
            df2 = pd.read_csv(file_metric)
            df2["basin_num"] = flag
            df2["basin_id"] = IDi
            df2["hru_id"] = IDi
            if df_metric.empty:
                df_metric = df2
            else:
                df_metric = pd.concat([df_metric, df2])
    
            flag += 1
    
        df_param.index = np.arange(len(df_param))
        
        df_metric.index = np.arange(len(df_metric))
        df_basinid = df_metric[["basin_num", "basin_id", "hru_id"]]
    
        selected_met = ['kge2009', 'r', 'alpha', 'beta', 'kge2012', 'rmse', 'me', 'mae', 'nse']
        df_metric = df_metric[selected_met]

        df_param.to_csv(file_all_param, index=False, compression='gzip')
        df_metric.to_csv(file_all_metric, index=False, compression='gzip')
        df_basinid.to_csv(file_all_basinid, index=False, compression='gzip')
        
    return df_param, df_metric, df_basinid


def load_allbasin_params_metrics_AllIter(inpath_basins, outpath_lse, gageIDs, iterend, external_iter0_info={}):
    # if len(external_iter0_path)>0, use iter-0 data from external_iter0_path
    
    # Load data: outputs from each iteration
    for iter in range(0, iterend):

        if iter==0 and len(external_iter0_info)>0:
            inpath_basins_iter0 = external_iter0_info['external_iter0_path']
            outpath_lse_iter0 = external_iter0_info['external_iter0_pathlse']
            df_param_i, df_metric_i, df_basinid_i = load_allbasin_params_metrics_OneIter(inpath_basins_iter0, outpath_lse_iter0, gageIDs, iter)
        else:
            df_param_i, df_metric_i, df_basinid_i = load_allbasin_params_metrics_OneIter(inpath_basins, outpath_lse, gageIDs, iter)
        
        df_basinid_i['iter'] = iter
        
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])

    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    df_basinid = df_basinid[~index]
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))

    return df_param, df_metric, df_basinid



def get_objective_function(df_metric, objfunc='normKGE2012'):

    if objfunc == 'normKGE2012':
        y = df_metric['kge2012'].values
        y = y / (2 - y)

    elif objfunc == 'normKGE2009':
        y = df_metric['kge2009'].values
        y = y / (2 - y)

    elif objfunc == 'normNSE':
        y = df_metric['nse'].values
        y = y / (2 - y)

    elif objfunc == 'truncNSE':
        y = df_metric['nse'].values.copy()
        y[y<-4] = -4

    elif objfunc == 'KGE2012':
        y = df_metric['kge2012'].values

    elif objfunc == 'KGE2009':
        y = df_metric['kge2009'].values

    elif objfunc == 'NSE':
        y = df_metric['nse'].values
    
    else:
        sys.exit(f'Not supported objfunc: {objfunc}')
    
    return y

# LSE calibration
##################

##################
# SSE calibration


def load_oneasin_params_metrics_AllIter(inpath_basins, gageID, iterend, external_iter0_info={}):
    # if len(external_iter0_path)>0, use iter-0 data from external_iter0_path
    
    # Load data: outputs from each iteration
    for iter in range(0, iterend):
        
        if iter==0 and len(external_iter0_info)>0:
            inpath_basins_iter0 = external_iter0_info['external_iter0_path']
            file_param = f"{inpath_basins_iter0}/{gageID}/param_sets/iter{iter}_all_param.csv"
            file_metric = f"{inpath_basins_iter0}/{gageID}/outputs/iter{iter}_all_metric.csv"
        else:
            file_param = f"{inpath_basins}/{gageID}/param_sets/iter{iter}_all_param.csv"
            file_metric = f"{inpath_basins}/{gageID}/outputs/iter{iter}_all_metric.csv"

        df_param_i = pd.read_csv(file_param)
        df_metric_i = pd.read_csv(file_metric)
        
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])

    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))

    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))

    return df_param, df_metric

# SSE calibration
##################



##################
# optimized algorithms


class MyProblem(Problem):
    def __init__(self, xlb_mean, xub_mean, em_model):
        super().__init__(n_var=len(xlb_mean), n_obj=1, n_constr=0, xl=xlb_mean, xu=xub_mean, elementwise_evaluation=False)
        self.em_model = em_model

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.em_model.predict(x)  # Return negative to maximize


def run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100, times=3):
    ga_all_solutions = []
    ga_all_outputs = []
    print(f'Run GA {times * num_runs} times and extract the best {num_runs} solutions')
    
    # Run the model `2 * num_runs` times
    for _ in range(times * num_runs):
        problem = MyProblem(xlb_mean, xub_mean, em_model)
        
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
            mutation=PolynomialMutation(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", num_generations),
                       verbose=False)
        
        optimized_features = res.X
        max_output = -res.F  # Assuming you want to maximize the output
        
        ga_all_solutions.append(optimized_features)
        ga_all_outputs.append(np.squeeze(max_output))
    
    # Convert to numpy arrays for easier sorting
    ga_all_solutions = np.array(ga_all_solutions)
    ga_all_outputs = np.array(ga_all_outputs)

    # Sort the results based on the max_output (in descending order)
    sorted_indices = np.argsort(ga_all_outputs)[::-1]  # Sort in descending order
    
    # Select the top `num_runs` solutions
    top_indices = sorted_indices[:num_runs]
    
    top_solutions = ga_all_solutions[top_indices]
    top_outputs = ga_all_outputs[top_indices]
    
    return top_solutions, top_outputs


def run_nsga2_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100):

    # perform optimization using the surrogate model
    pop = 100
    if pop <= num_runs:
        pop = num_runs * 2
    
    gen = 100
    crossover_rate = 0.9
    mu = 20
    mum = 20
    bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(em_model, len(xlb_mean), 2, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
    D = NSGA2.crowding_distance(besty_sm)
        
    idxr = D.argsort()[::-1][:num_runs]
    nsga2_all_solutions = bestx_sm[idxr, :]
    nsga2_all_outputs = besty_sm[idxr, :]

    return np.array(nsga2_all_solutions), np.array(nsga2_all_outputs)


def generate_optimized_paramsets(outpath, iterend, em_model, param_lb, param_ub, param_calibflag, param_names, numruns, num_objfunc=1, times=3):
    # param_calibflag (vector): 1 means calibrate parameters (different lb and ub); 0 means non calibrated param (same lb and ub)
    
    # generate optimized parameter sets
    outfile_optmz = f'{outpath}/optmz_output_iter{iterend}.npz'
    os.makedirs(outpath, exist_ok=True)
    if os.path.isfile(outfile_optmz):
        dtmp = np.load(outfile_optmz)
        optmz_param_all = dtmp['optmz_param_all']
        optmz_target_all = dtmp['optmz_target_all']
        
    else:
        if num_objfunc == 1:
            optmz_param_all, optmz_target_all = run_ga_optimization(em_model, param_lb, param_ub, num_runs=numruns, pop_size=100, num_generations=100, times=times)
            optmz_param_all = np.array(optmz_param_all)
    
        elif num_objfunc == 2:
            optmz_param_all, optmz_target_all = run_nsga2_optimization(em_model, param_lb, param_ub, num_runs=numruns, pop_size=100, num_generations=100)
            optmz_param_all = np.array(optmz_param_all)
    
        np.savez_compressed(outfile_optmz, optmz_param_all=optmz_param_all, optmz_target_all=optmz_target_all)

    
    for i in range(optmz_param_all.shape[0]):
        outfile = f'{outpath}/paramset_iter{iterend}_trial{i}.pkl'
        if os.path.isfile(outfile):
            continue

        dfi = pd.DataFrame()
        dfi['Parameter'] = param_names[param_calibflag==1]
        dfi['Value'] = optmz_param_all[i, param_calibflag==1]

        dfi.to_pickle(outfile)



# optimized algorithms
##################
