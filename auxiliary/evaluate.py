import numpy as np
import pandas as pd
import sys, glob, os, subprocess

def get_KGE(obs, sim):
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge, r, alpha, beta

def get_modified_KGE(obs,sim):
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]

    try:
        sd_sim=np.std(sim, ddof=1)
        sd_obs=np.std(obs, ddof=1)
        m_sim=np.mean(sim)
        m_obs=np.mean(obs)
        r=(np.corrcoef(sim,obs))[0,1]
        relvar=(float(sd_sim)/float(m_sim))/(float(sd_obs)/float(m_obs))
        bias=float(m_sim)/float(m_obs)
        kge=1.0-np.sqrt((r-1)**2 +(relvar-1)**2 + (bias-1)**2)
    except:
        kge = np.nan

    return kge

def get_CC(obs,sim):
    cc = np.corrcoef(obs, sim)[0, 1]
    return cc

def get_RMSE(obs,sim):
    rmse = np.sqrt(np.nanmean(np.power((sim - obs),2)))
    return rmse

def get_mean_error(obs, sim):
    me = np.nanmean(sim - obs)
    return me

def get_mean_abs_error(obs, sim):
    mean_abs_err = np.nanmean(np.absolute(sim - obs))
    return mean_abs_err

def get_nse(obs, sim):   
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]
    
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = (np.mean(sim) - np.mean(obs)) / np.std(obs)
    
    nse = 1-(np.nansum((sim-obs)**2)/np.nansum((obs-np.nanmean(obs))**2))

    return nse, r, alpha, beta

def cal_metrics(obs, sim, outfile_metric='', overwrite=True):

    if len(outfile_metric)>0:
        if os.path.isfile(outfile_metric) and overwrite==False:
            print('Metric file exists:', outfile_metric)
            dfout = pd.read_csv(outfile_metric)
            return dfout
    
    # deal with nan
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]

    # calculate metrics
    kge2009, r_kge, alpha_kge, beta_kge = get_KGE(obs, sim)
    kge2012 = get_modified_KGE(obs,sim)
    rmse = get_RMSE(obs,sim)
    me = get_mean_error(obs, sim)
    mae = get_mean_abs_error(obs, sim)
    nse, r_nse, alpha_nse, beta_nse = get_nse(obs, sim)

    dfout = pd.DataFrame([[kge2009, r_kge, r_nse, alpha_kge, alpha_nse, beta_kge, beta_nse, kge2012, rmse, me, mae, nse]], 
                         columns=['kge2009', 'r (kge)', 'r (nse)', 'alpha (kge)', 'alpha (nse)', 'beta (kge)', 'beta (nse)', 'kge2012', 'rmse', 'me', 'mae', 'nse'])

    if len(outfile_metric)>0:
        dfout.to_csv(outfile_metric, index=False)

    return dfout