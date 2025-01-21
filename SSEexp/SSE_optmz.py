# functions for generating and saving MO-ASMO parameter sets, including initial and non-dominant parameters
import os, sys, subprocess, pickle, random
import numpy as np
import pandas as pd
import xarray as xr
import pickle

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from mo_evaluation import get_modified_KGE

path_MOASMO = '/glade/u/home/guoqiang/CTSM_repos/ctsm_optz/MO-ASMO/src/'
sys.path.append(path_MOASMO)
import sampling
import gp
import NSGA2

##############
# single objective optimization

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination

class MyProblem(Problem):
    def __init__(self, xlb_mean, xub_mean, em_model):
        super().__init__(n_var=len(xlb_mean), n_obj=1, n_constr=0, xl=xlb_mean, xu=xub_mean, elementwise_evaluation=False)
        self.em_model = em_model

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.em_model.predict(x)  # Return negative to maximize


def run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100):
    ga_all_solutions = []
    ga_all_outputs = []

    for i in range(num_runs):  # run the model `num_runs` times
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
        max_output = -res.F
        
        ga_all_solutions.append(optimized_features)
        ga_all_outputs.append(np.squeeze(max_output))
    
    return np.array(ga_all_solutions), np.array(ga_all_outputs)



########################################################################################################################
# Pareto optimal points:

def get_rmse(d1, d2):
    return ( np.nanmean( (d1-d2)**2 ) ) ** 0.5

def gpr_emulator_cv(x, y, alpha, leng_lb, leng_ub, nu, xlb_mean, xub_mean, outpath, iterflag, rndseed=1234567890, usemet='kge'):

    random.seed(rndseed)
    np.random.seed(rndseed)

    n_splits = 5
    
    kf = KFold(n_splits=n_splits, shuffle=True) 
    kge_scores = np.nan * np.zeros([n_splits, y.shape[1]])
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x), 1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize and train your GPR model here; adjust parameters as needed
        sm = gp.GPR_Matern(x_train, y_train, x_train.shape[1], y_train.shape[1], x_train.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
        
        # Predict using the trained model
        y_pred = sm.predict(x_test)  # Adjust this method call based on your model's API
        
        # Evaluate the model using KGE
        for i in range(y.shape[1]):
            if usemet == 'kge':
                kge_scores[fold_idx-1, i] = get_modified_KGE(y_test[:,i], y_pred[:,i])
            elif usemet == 'rmse': # temporary codes 
                kge_scores[fold_idx-1, i] = get_rmse(y_test[:,i], y_pred[:,i])
            else:
                sys.exit('Unknown metric')
    
    # Calculate the mean KGE score across all folds
    mean_kge_score = np.nanmean(kge_scores, axis=0)[np.newaxis, :]
    kge_scores = np.concatenate([kge_scores, mean_kge_score])

    # Convert the list of KGE scores into a pandas DataFrame
    kge_scores_df = pd.DataFrame()
    kge_scores_df['Fold'] = list(np.arange(n_splits)+1) + ['mean']
    kge_scores_df[f'{usemet}1'] = kge_scores[:, 0]

    if kge_scores.shape[1]>1:
        kge_scores_df[f'{usemet}2'] = kge_scores[:, 1]
        kge_scores_df[f'{usemet}_mean'] = (kge_scores[:, 0] + kge_scores[:, 1])/2
    
    print("GPR CV KGE Score for metric1/metric2:")
    print(kge_scores_df)
    
    csv_file_path =  f'{outpath}/GPR_for_iter{iterflag}_CV_kge.csv'
    kge_scores_df.to_csv(csv_file_path, index=False)

    return kge_scores_df


def rf_emulator_cv(x, y, outpath, iterflag, usemet='kge'):

    random.seed(1234567890)
    np.random.seed(1234567890)
    
    n_splits = 5
    
    kf = KFold(n_splits=n_splits, shuffle=True) 
    kge_scores = np.nan * np.zeros([n_splits, y.shape[1]])
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x), 1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize and train your GPR model here; adjust parameters as needed
        sm = RandomForestRegressor()
        sm.fit(x_train, y_train)
        
        # Predict using the trained model
        y_pred = sm.predict(x_test)  # Adjust this method call based on your model's API

        if y.shape[1] == 1:
            y_pred = y_pred[:, np.newaxis]
        
        # Evaluate the model using KGE
        for i in range(y.shape[1]):
            if usemet == 'kge':
                kge_scores[fold_idx-1, i] = get_modified_KGE(y_test[:,i], y_pred[:,i])
            elif usemet == 'rmse': # temporary codes 
                kge_scores[fold_idx-1, i] = get_rmse(y_test[:,i], y_pred[:,i])
            else:
                sys.exit('Unknown metric')
    
    # Calculate the mean KGE score across all folds
    mean_kge_score = np.nanmean(kge_scores, axis=0)[np.newaxis, :]
    kge_scores = np.concatenate([kge_scores, mean_kge_score])

    # Convert the list of KGE scores into a pandas DataFrame
    kge_scores_df = pd.DataFrame()
    kge_scores_df['Fold'] = list(np.arange(n_splits)+1) + ['mean']
    kge_scores_df[f'{usemet}1'] = kge_scores[:, 0]

    if kge_scores.shape[1]>1:
        kge_scores_df[f'{usemet}2'] = kge_scores[:, 1]
        kge_scores_df[f'{usemet}_mean'] = (kge_scores[:, 0] + kge_scores[:, 1])/2
        
    print("RF CV KGE Score for metric1/metric2:")
    print(kge_scores_df)
    
    csv_file_path =  f'{outpath}/RF_for_iter{iterflag}_CV_kge.csv'
    kge_scores_df.to_csv(csv_file_path, index=False)

    return kge_scores_df


def surrogate_model_train_and_optmz(x, y, xlb_mean, xub_mean, param_names, outpath, iterflag, num_per_iter, normalize_y=False):

    random.seed(1234567890)
    print('Normalize_y', normalize_y)

    onlyuse_GPR = True

    # check whether files have been generated
    flag = False
    for i in range(num_per_iter):
        outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.pkl'
        if not os.path.isfile(outfile):
            flag = True
            break
            
    if flag == False:
        print('All parameter pickle files have been generated. Skip this step')
    else:
        # define hyper parameters
        pop = 100
        gen = 100
        crossover_rate = 0.9
        mu = 20
        mum = 20

        # define hyperparameter
        alpha = 1e-3
        leng_lb = 1e-3
        leng_ub = 1e3
        nu = 2.5

        n_sample = num_per_iter # number of selected optimal points

        # in / out
        ind = ~np.isnan( np.sum(x,axis=1) + np.sum(y,axis=1))
        x, y = x[ind, :], y[ind, :]

        if normalize_y == True:
            for i in range(y.shape[1]):
                y[:,i] = (y[:,i] - np.nanmin(y[:,i])) / (np.nanmax(y[:, i]) - np.nanmin(y[:,i]))

        nInput = x.shape[1]
        nOutput = y.shape[1]


        # decide the most suitable emulator based on cross validation
        os.makedirs(outpath, exist_ok=True)
        gpr_kge_cv = gpr_emulator_cv(x, y, alpha, leng_lb, leng_ub, nu, xlb_mean, xub_mean, outpath, iterflag, usemet='rmse')
        rf_kge_cv = rf_emulator_cv(x, y, outpath, iterflag, usemet='rmse')

        # train the surrogate model 
        if gpr_kge_cv['rmse1'].values[-1] < rf_kge_cv['rmse1'].values[-1]:
        # if True: # always use GPR
            print('Use GPR model')
            nInput = x.shape[0]
            nOutput = 1
            sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
            flag = 1
        else:
            print('Use RF model')
            sm = RandomForestRegressor()
            sm.fit(x, y)
            flag = 2


        # perform optimization using the surrogate model
        x_resample, y_resample = run_ga_optimization(sm, xlb_mean, xub_mean, num_runs=n_sample, pop_size=60, num_generations=60) # for larger pop_size/num_generations, outputs from different runs could be similar


        for i in range(x_resample.shape[0]):
            outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.pkl'
            if os.path.isfile(outfile):
                continue
    
            dfi = pd.DataFrame()
            dfi['Parameter'] = param_names
            dfi['Value'] = x_resample[i, :]
    
            dfi.to_pickle(outfile)

        # save intermediate outputs for check
        # outfile = f'{outpath}/intermediate_output_iter{iterflag+1}.pkl'
        # np.savez_compressed(outfile, x_resample=x_resample, y_resample=y_resample, xlb_mean=xlb_mean, xub_mean=xub_mean, x=x, y=y)
