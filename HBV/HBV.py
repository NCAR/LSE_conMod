# HBV model changed from hydroDL2
# Source: https://github.com/mhpi/hydroDL2/blob/master/src/hydroDL2/models/hbv/hbv_1_1p.py
# Created by Guoqiang Tang on Dec 16, 2024

import pickle, os
import numpy as np
import pandas as pd
from scipy.special import gamma 

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
        Input forcing data P/T/PET [time].
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

    n_steps = len(P)

    # Initialize time series of model variables
    Qsim = np.zeros((n_steps), dtype=np.float32) + 0.001
    Q0_sim = np.zeros((n_steps), dtype=np.float32) + 0.0001
    Q1_sim = np.zeros((n_steps), dtype=np.float32) + 0.0001
    Q2_sim = np.zeros((n_steps), dtype=np.float32) + 0.0001

    AET = np.zeros((n_steps), dtype=np.float32)
    recharge_sim = np.zeros((n_steps), dtype=np.float32)
    excs_sim = np.zeros((n_steps), dtype=np.float32)
    evapfactor_sim = np.zeros((n_steps), dtype=np.float32)
    tosoil_sim = np.zeros((n_steps), dtype=np.float32)
    PERC_sim = np.zeros((n_steps), dtype=np.float32)
    SWE_sim = np.zeros((n_steps), dtype=np.float32)
    capillary_sim = np.zeros((n_steps), dtype=np.float32)

    for t in range(n_steps):

        # Separate precipitation into liquid and solid components
        PRECIP = P[t]
        RAIN = PRECIP * (T[t] >= param_dict['parTT']).astype(np.float32)
        SNOW = PRECIP * (T[t] < param_dict['parTT']).astype(np.float32)

        # Snow
        SNOWPACK = SNOWPACK + SNOW
        melt = param_dict['parCFMAX'] * (T[t] - param_dict['parTT'])
        melt = np.clip(melt, 0, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = param_dict['parCFR'] * param_dict['parCFMAX'] * (param_dict['parTT'] - T[t])
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
        ETact = PET[t] * evapfactor
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
        Qsim[t] = Q0 + Q1 + Q2
        Q0_sim[t] = Q0
        Q1_sim[t] = Q1
        Q2_sim[t] = Q2
        AET[t] = ETact
        SWE_sim[t] = SNOWPACK
        capillary_sim[t] = capillary

        recharge_sim[t] = recharge
        excs_sim[t] = excess
        evapfactor_sim[t] = evapfactor
        tosoil_sim[t] = tosoil
        PERC_sim[t] = PERC

    # Run routing
    if routing:
        # Generate unit hydrograph
        UH = UH_gamma(routing_params['rout_a'], routing_params['rout_b'], lenF=15)
    
        # Perform routing
        Qsrout = UH_conv(Qsim, UH)
    
        # Routing individually for Q0, Q1, and Q2
        Q0_rout = UH_conv(Q0_sim, UH)
        Q1_rout = UH_conv(Q1_sim, UH)
        Q2_rout = UH_conv(Q2_sim, UH)

        Qs = Qsrout

    else:
        Qs = Qsim
        Q0_rout = Q1_rout = Q2_rout = None

    if initialize:
        # If initialize is True, only return warmed-up storages.
        return SNOWPACK, MELTWATER, SM, SUZ, SLZ
    else:
        # Baseflow index (BFI) calculation
        BFI_sim = 100 * (np.sum(Q2_sim) / (np.sum(Qsim) + nearzero))
    
        # Return all simulation results.
        out_dict = {
            'flow_sim': Qs, # unit: mm/day
            'srflow': Q0_rout,
            'ssflow': Q1_rout,
            'gwflow': Q2_rout,
            'AET_hydro': AET,
            'PET_hydro': PET,
            'SWE': SWE_sim,
            'flow_sim_no_rout': Qsim,
            'srflow_no_rout': Q0_sim,
            'ssflow_no_rout': Q1_sim,
            'gwflow_no_rout': Q2_sim,
            'recharge': recharge_sim,
            'excs': excs_sim,
            'evapfactor': evapfactor_sim,
            'tosoil': tosoil_sim,
            'percolation': PERC_sim,
            'capillary': capillary_sim,
            'BFI_sim': BFI_sim
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
