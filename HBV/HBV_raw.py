# HBV model changed from hydroDL2
# Source: https://github.com/mhpi/hydroDL2/blob/master/src/hydroDL2/models/hbv/hbv_1_1p.py
# Created by Guoqiang Tang on Dec 16, 2024

import pickle, os
import numpy as np
import pandas as pd
from scipy.special import gamma 

def UH_gamma(a, b, lenF=10):
    # UH. a [time (same all time steps), batch, var]
    lenF = min(a.shape[0], lenF)
    t = np.arange(0.5, lenF, 1).reshape((lenF, 1, 1))
    aa = np.maximum(a[:lenF, :, :], 0.1)  # Minimum value of 0.1
    theta = np.maximum(b[:lenF, :, :], 0.5)  # Minimum value of 0.5

    denom = gamma(aa) * (theta ** aa)  # Use scipy.special.gamma for arrays
    mid = t ** (aa - 1)
    right = np.exp(-t / theta)
    w = (1 / denom) * mid * right
    w = w / w.sum(axis=0)  # Normalize to 1 for each hydrograph
    return w


def UH_conv(x, UH, viewmode=1):
    
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view

    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html
    
    nb, var, time = x.shape
    uhLen = UH.shape[-1]
    padd = uhLen - 1

    if viewmode == 1:
        xx = x.reshape((1, nb * var, time))
        w = UH.reshape((nb * var, 1, uhLen))
    else:
        raise ValueError("Unsupported viewmode")

    Qr = np.zeros((xx.shape[0], xx.shape[1], xx.shape[2] + padd))
    for i in range(xx.shape[1]):
        Qr[:, i, :] = np.convolve(xx[0, i, :], w[i, 0, ::-1], mode='full')

    Qr = Qr[:, :, :time]  # Remove extra padding

    return Qr.reshape((nb, var, time))


def load_HBV_param(infile_param):
    # read HBV parameters from a pickle file

    try:
        df = pd.read_pickle(infile_param)
    except:
        df = pd.read_csv(infile_param)
    
    param1 = ['parBETA', 'parFC', 'parK0', 'parK1', 'parK2', 'parLP', 'parPERC', 'parUZL', 'parTT', 'parCFMAX', 'parCFR', 'parCWH', 'parBETAET', 'parC']
    param2 = ['rout_a', 'rout_b']

    param1_dict = {}
    for p in param1:
        param1_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    param2_dict = {}
    for p in param2:
        param2_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    return param1_dict, param2_dict


def HBV_np(P, T, PET, states, param_dict, routing_params, routing=True, initialize=False, muwts=None, nearzero=1e-5):
    """
    Run the HBV1.1p model forward in NumPy.

    Parameters
    ----------
    forcing : np.ndarray
        Input forcing data P/T/PET [time, grid].
    states : tuple
        Initial model states (SNOWPACK, MELTWATER, SM, SUZ, SLZ).
    param_dict : dict
        Dictionary of model parameters. it contains one value for each param and does not use the dynamic param as in full_param_dict
    variables : list
        List of variable names corresponding to forcing data.
    routing_params : dict
        Routing parameters containing 'rout_a' and 'rout_b'.
    routing : bool, optional
        Whether to perform routing, by default True.
    nearzero : float, optional
        Minimum value to avoid division by zero, by default 1e-5.

    Returns
    -------
    dict
        Dictionary of model outputs.
    """
    
    SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

    # # Forcings
    # P = forcing[:, :, variables.index('prcp')]  # Precipitation
    # T = forcing[:, :, variables.index('tmean')]  # Mean air temp
    # PET = forcing[:, :, variables.index('pet')]  # Potential ET

    n_steps, n_grid = P.shape

    # Initialize time series of model variables
    Qsimmu = np.zeros((n_steps, n_grid), dtype=np.float32) + 0.001
    Q0_sim = np.zeros((n_steps, n_grid), dtype=np.float32) + 0.0001
    Q1_sim = np.zeros((n_steps, n_grid), dtype=np.float32) + 0.0001
    Q2_sim = np.zeros((n_steps, n_grid), dtype=np.float32) + 0.0001

    AET = np.zeros((n_steps, n_grid), dtype=np.float32)
    recharge_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    excs_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    evapfactor_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    tosoil_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    PERC_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    SWE_sim = np.zeros((n_steps, n_grid), dtype=np.float32)
    capillary_sim = np.zeros((n_steps, n_grid), dtype=np.float32)

    for t in range(n_steps):
        # (not used here) Get dynamic parameter values per timestep
        # for key in full_param_dict.keys():
        #     param_dict[key] = full_param_dict[key][t, :, :]

        # Separate precipitation into liquid and solid components
        PRECIP = P[t, :]
        RAIN = PRECIP * (T[t, :] >= param_dict['parTT']).astype(np.float32)
        SNOW = PRECIP * (T[t, :] < param_dict['parTT']).astype(np.float32)

        # Snow
        SNOWPACK = SNOWPACK + SNOW
        melt = param_dict['parCFMAX'] * (T[t, :] - param_dict['parTT'])
        melt = np.clip(melt, 0, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = param_dict['parCFR'] * param_dict['parCFMAX'] * (param_dict['parTT'] - T[t, :])
        refreezing = np.clip(refreezing, 0, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing
        tosoil = np.clip(MELTWATER - param_dict['parCWH'] * SNOWPACK, 0, None)
        MELTWATER = MELTWATER - tosoil

        # Soil and evaporation
        soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
        soil_wetness = np.clip(soil_wetness, 0, 1)
        recharge = (RAIN + tosoil) * soil_wetness

        SM = SM + RAIN + tosoil - recharge

        excess = SM - param_dict['parFC']
        excess = np.clip(excess, 0, None)
        SM = SM - excess
        # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
        evapfactor = (SM / (param_dict['parLP'] * param_dict['parFC'])) ** param_dict['parBETAET']
        evapfactor = np.clip(evapfactor, 0, 1)
        ETact = PET[t, :] * evapfactor
        ETact = np.minimum(SM, ETact)
        SM = np.clip(SM - ETact, nearzero, None)

        # Capillary rise
        capillary = np.minimum(SLZ, param_dict['parC'] * SLZ * (1.0 - np.clip(SM / param_dict['parFC'], 0, 1)))
        SM = np.clip(SM + capillary, nearzero, None)
        SLZ = np.clip(SLZ - capillary, nearzero, None)

        # Groundwater boxes
        SUZ = SUZ + recharge + excess
        PERC = np.minimum(SUZ, param_dict['parPERC'])
        SUZ = SUZ - PERC
        Q0 = param_dict['parK0'] * np.clip(SUZ - param_dict['parUZL'], 0, None)
        SUZ = SUZ - Q0
        Q1 = param_dict['parK1'] * SUZ
        SUZ = SUZ - Q1
        SLZ = SLZ + PERC
        Q2 = param_dict['parK2'] * SLZ
        SLZ = SLZ - Q2

        # Save results
        Qsimmu[t, :] = Q0 + Q1 + Q2
        Q0_sim[t, :] = Q0
        Q1_sim[t, :] = Q1
        Q2_sim[t, :] = Q2
        AET[t, :] = ETact
        SWE_sim[t, :] = SNOWPACK
        capillary_sim[t, :] = capillary

        recharge_sim[t, :] = recharge
        excs_sim[t, :] = excess
        evapfactor_sim[t, :] = evapfactor
        tosoil_sim[t, :] = tosoil
        PERC_sim[t, :] = PERC

    # Get the overall average 
    # or weighted average using learned weights.
    if Qsimmu.ndim == 3:
        if muwts is None:  
            Qsimavg = Qsimmu.mean(axis=-1) # dim: [time, n_grid]
        else:
            Qsimavg = (Qsimmu * muwts).sum(axis=-1) 
    else:
        Qsimavg = Qsimmu
        
    # Run routing
    if routing:
        # (not supported here) Routing for all components or just the average.
        # if comprout:
        #     # All components; reshape to [time, gages * num models]
        #     Qsim = Qsimmu.reshape(n_steps, n_grid * nmul)
        # else:
        #     # Average, then do routing.
        #     Qsim = Qsimavg
        
        Qsim = Qsimavg
    
        # Generate unit hydrograph
        uh_a = np.zeros([n_steps, 1, 1], dtype=np.float32)
        uh_a[:] = routing_params['rout_a']
        uh_b = np.zeros([n_steps, 1, 1], dtype=np.float32)
        uh_b[:] = routing_params['rout_b']
        UH = UH_gamma(uh_a, uh_b, lenF=15)
    
        # Perform routing for all components
        rf = Qsim[:, :, np.newaxis].transpose(1, 2, 0)  # [gages, vars, time]
        UH = UH.transpose(1, 2, 0)  # [gages, vars, time]
        Qsrout = UH_conv(rf, UH).transpose(2, 0, 1)
    
        # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time]
        # rf_Q0 = Q0_sim.mean(axis=-1, keepdims=True).transpose(1, 2, 0)
        rf_Q0 = Q0_sim[:,:,np.newaxis].transpose(1, 2, 0)
        Q0_rout = UH_conv(rf_Q0, UH).transpose(2, 0, 1)
        # rf_Q1 = Q1_sim.mean(axis=-1, keepdims=True).transpose(1, 2, 0)
        rf_Q1 = Q1_sim[:,:,np.newaxis].transpose(1, 2, 0)
        Q1_rout = UH_conv(rf_Q1, UH).transpose(2, 0, 1)
        # rf_Q2 = Q2_sim.mean(axis=-1, keepdims=True).transpose(1, 2, 0)
        rf_Q2 = Q2_sim[:,:,np.newaxis].transpose(1, 2, 0)
        Q2_rout = UH_conv(rf_Q2, UH).transpose(2, 0, 1)

        # not supported
        # if comprout:
        #     # Qs is now shape [time, gages, num models]
        #     Qstemp = Qsrout.reshape(n_steps, n_grid, nmul)
        #     if muwts is None:
        #         Qs = Qstemp.mean(axis=-1, keepdims=True)
        #     else:
        #         Qs = (Qstemp * muwts).sum(axis=-1, keepdims=True)
        # else:
        #     Qs = Qsrout
        Qs = Qsrout

    else:
        # No routing, only output the average of all model sims.
        Qs = Qsimavg[:, :, np.newaxis]
        Q0_rout = Q1_rout = Q2_rout = None

    if initialize:
        # If initialize is True, only return warmed-up storages.
        return SNOWPACK, MELTWATER, SM, SUZ, SLZ
    else:
        # Baseflow index (BFI) calculation
        BFI_sim = 100 * (np.sum(Q2_sim, axis=0) / (np.sum(Qsimmu, axis=0) + nearzero))
    
        # Return all simulation results.
        out_dict = {
            'flow_sim': Qs.squeeze(), # unit: mm/day
            'srflow': Q0_rout.squeeze(),
            'ssflow': Q1_rout.squeeze(),
            'gwflow': Q2_rout.squeeze(),
            'AET_hydro': np.mean(AET, axis=-1, keepdims=True).squeeze(),
            'PET_hydro': np.mean(PET, axis=-1, keepdims=True).squeeze(),
            'SWE': np.mean(SWE_sim, axis=-1, keepdims=True).squeeze(),
            'flow_sim_no_rout': np.expand_dims(Qsim, axis=2).squeeze(),
            'srflow_no_rout': np.mean(Q0_sim, axis=-1, keepdims=True).squeeze(),
            'ssflow_no_rout': np.mean(Q1_sim, axis=-1, keepdims=True).squeeze(),
            'gwflow_no_rout': np.mean(Q2_sim, axis=-1, keepdims=True).squeeze(),
            'recharge': np.mean(recharge_sim, axis=-1, keepdims=True).squeeze(),
            'excs': np.mean(excs_sim, axis=-1, keepdims=True).squeeze(),
            'evapfactor': np.mean(evapfactor_sim, axis=-1, keepdims=True).squeeze(),
            'tosoil': np.mean(tosoil_sim, axis=-1, keepdims=True).squeeze(),
            'percolation': np.mean(PERC_sim, axis=-1, keepdims=True).squeeze(),
            'capillary': np.mean(capillary_sim, axis=-1, keepdims=True).squeeze(),
            'BFI_sim': BFI_sim.squeeze()
        }

        return out_dict

def save_HBV_output(filename, data, overwrite=True):
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

def load_HBV_output(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def run_HBV_OneBasin_OneParamSet(P, T, PET, file_paramset, outfile ):
    # run HBV on one parameter set

    if os.path.isfile(outfile):
        print('HBV output file exists:', outfile)
        print('Skip HBV modeling')

    else:
        # print('Run HBV using cold initial states')

        HBVparam_dict, routingparam_dict = load_HBV_param(file_paramset)
    
        SNOWPACK = np.zeros([1,1], dtype=np.float32) + 0.001
        MELTWATER = np.zeros([1,1], dtype=np.float32) + 0.001
        SM = np.zeros([1,1], dtype=np.float32) + 0.001
        SUZ = np.zeros([1,1], dtype=np.float32) + 0.001
        SLZ = np.zeros([1,1], dtype=np.float32) + 0.001
        coldstates = (SNOWPACK, MELTWATER, SM, SUZ, SLZ)
    
        HBV_simu = HBV_np(P, T, PET, coldstates, HBVparam_dict, routingparam_dict)

        print('Saving output file to', outfile)
        save_HBV_output(outfile, HBV_simu)
