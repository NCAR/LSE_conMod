# source of translation: https://github.com/VT-Hydroinformatics/16-Intro-Modeling-HBV/blob/master/HBV/HBV.R
# created by Guoqiang Tang

## original comments
# HBV (Hydrologiska byrÂns vattenavdelning) model 
# Code for initializing HBV hydrologic simulation model
#
# References:
# Bergstrˆm, S., & Forsman, A. (1973). Development of a conceptual 
#     deterministic rainfall-runoff model. Nordic Hydrology, 4(3), 147-170.
# Bergstrˆm, S. (1995) The HBV model (Chapter 13, pp. 443-476), in: 
#     Singh, V.P. (ed.) Computer models of watershed hydrology, Water 
#     Resources Publications, Highlands Ranch, Colorado, U.S.A., 1130 pp.
# Seibert, J. (2002) HBV light version 2, User's manual, 
#     Environmental Assessment, SLU, Uppsala
#
# Matlab code written by K. McGuire, March, 2012. 
# Thanks to J. Seibert for sharing parts of HBV Light
# Code written by K. McGuire, March 2012
# Translated to R by J.P. Gannon, March 2021

# HBV (Hydrologiska byrÂns vattenavdelning) model 
#   (version does not include elevation/vegetation zones)
#
# INPUTS:
#   pars    = parameters (listed below in order)
#   P       = precipitation daily time series, mm/d
#   T       = mean daily temperature, deg C
#   PET     = estimate of daily potential evapotranspiration, mm/d
#   routing = when this term = 1, then triangular routing is invoked, or
#             routing = 0 
#
# OUTPUTS:
#   q       = simulated streamflow, mm/d
#   qs      = simulated shallow flow (e.g., SOF, macropores, etc.), mm/d
#   qi      = simulated interflow (i.e., from upper GW storage), mm/d
#   qb      = simulated baseflow (i.e., from lower GW storage), mm/d
#   Storage = simulated total storage (= soil + GW1 + GW2), mm
#   SWE     = simulated snow water equivalent, mm
#   AET     = simulated actual evapotranspiration, mm/d
#   SF      = simulated snowfall (includes evap and catch efficiency), mm/d
#   S1      = simulated storage from the upper GW reservoir, mm
#   S2      = simulated storage from the lower GW reservoir, mm
#   soil    = simulated storage in soil, mm
#   w       = simulated water inputs to soil (= rainfall + snowmelt), mm/d



import numpy as np
import pandas as pd
import pickle, os

def HBV_KM(P, T, PET, param_dict, routing_params, routing=1):
    """
    HBV hydrological model implementation in Python.

    Parameters:
        param_dict: dict
            Model parameters including:
                - FC: Max soil moisture storage, field capacity.
                - beta: Shape coefficient governing fate of water input to soil moisture storage.
                - LP: Threshold for reduction of evapotranspiration.
                - SFCF: Snowfall correction factor.
                - TT: Threshold temperature.
                - CFMAX: Degree-day factor.
                - k0: Recession constant (upper storage, near surface).
                - k1: Recession constant (upper storage).
                - k2: Recession constant (lower storage).
                - UZL: Threshold for shallow storage.
                - PERC: Percolation, max flow from upper to lower storage.
        routing_params: dict
            Routing parameters including:
                - MAXBAS: Maximum basin time for triangular routing.
        P: np.ndarray
            Daily precipitation (mm/d).
        T: np.ndarray
            Daily mean temperature (°C).
        PET: np.ndarray
            Daily potential evapotranspiration (mm/d).
        routing: int
            Set to 1 to enable triangular routing, 0 otherwise.

    Returns:
        dict
            Contains simulated streamflow and associated variables.
    """
    # Extract parameters
    FC = param_dict['FC']  # Max soil moisture storage, field capacity.
    beta = param_dict['beta']  # Shape coefficient governing fate of water input to soil moisture storage.
    LP = param_dict['LP']  # Threshold for reduction of evapotranspiration.
    SFCF = param_dict['SFCF']  # Snowfall correction factor.
    TT = param_dict['TT']  # Threshold temperature.
    CFMAX = param_dict['CFMAX']  # Degree-day factor.
    k0 = param_dict['k0']  # Recession constant (upper storage, near surface).
    k1 = param_dict['k1']  # Recession constant (upper storage).
    k2 = param_dict['k2']  # Recession constant (lower storage).
    UZL = param_dict['UZL']  # Threshold for shallow storage.
    PERC = param_dict['PERC']  # Percolation, max flow from upper to lower storage.
    MAXBAS = routing_params['MAXBAS']  # Maximum basin time for triangular routing.

    CFR = 0.05  # Refreezing coefficient (fixed value).
    CWH = 0.1   # Water holding capacity (fixed value).

    n_days = len(P)

    # Initialize arrays for outputs
    AET = np.zeros(n_days)  # Actual evapotranspiration, mm/d.
    SF = np.zeros(n_days)  # Snowfall, mm.
    R = np.zeros(n_days)  # Recharge, mm/d.
    soil = np.zeros(n_days)  # Soil storage, mm.
    SWE = np.zeros(n_days)  # Snow water equivalent, mm.
    w = np.zeros(n_days)  # Water input = snowmelt + rainfall, mm/d.
    Storage = np.zeros(n_days)  # Total storage, mm.
    S1 = np.zeros(n_days)  # Upper zone storage, mm.
    S2 = np.zeros(n_days)  # Lower zone storage, mm.
    Q_STZ = np.zeros(n_days)  # Shallow flow (saturated overland flow), mm/d.
    Q_SUZ = np.zeros(n_days)  # Upper zone flow (interflow), mm/d.
    Q_SLZ = np.zeros(n_days)  # Lower zone flow (baseflow), mm/d.
    Qgen = np.zeros(n_days)  # Total generated streamflow, mm/d.

    # Initialize states
    SP = 152.4  # Initial snowpack, mm.
    WC = 0.0    # Initial liquid water in snowpack, mm.
    SM = FC     # Initial soil moisture, mm.
    SUZ = 0.0   # Initial upper zone storage, mm.
    SLZ = 0.0   # Initial lower zone storage, mm.

    # Time loop
    for t in range(1, n_days):
        # Snow accumulation and melt
        inc = 0.0
        if SP > 0:
            if P[t] > 0:
                if T[t] > TT:
                    WC += P[t]
                else:
                    SP += P[t] * SFCF

            if T[t] > TT:
                melt = CFMAX * (T[t] - TT)
                if melt > SP:
                    inc = SP + WC
                    WC = 0.0
                    SP = 0.0
                else:
                    SP -= melt
                    WC += melt
                    if WC >= CWH * SP:
                        inc = WC - CWH * SP
                        WC = CWH * SP
            else:
                refreeze = CFR * CFMAX * (TT - T[t])
                if refreeze > WC:
                    refreeze = WC
                SP += refreeze
                WC -= refreeze
                SF[t] = P[t] * SFCF
        else:
            if T[t] > TT:
                inc = P[t]
            else:
                SP += P[t] * SFCF

        SWE[t] = SP + WC

        # Soil moisture
        rQ = 0.0
        old_SM = SM
        if inc > 0:
            if inc < 1:
                y = inc
            else:
                m = int(np.floor(inc))
                y = inc - m
                for _ in range(m):
                    dQdP = (SM / FC) ** beta
                    if dQdP > 1:
                        dQdP = 1
                    SM += 1 - dQdP
                    rQ += dQdP

            dQdP = (SM / FC) ** beta
            if dQdP > 1:
                dQdP = 1
            SM += (1 - dQdP) * y
            rQ += dQdP * y

        mean_SM = (SM + old_SM) / 2
        if mean_SM < LP * FC:
            AET[t] = PET[t] * mean_SM / (LP * FC)
        else:
            AET[t] = PET[t]

        if SP + WC > 0:
            AET[t] = 0
        SM -= AET[t]

        if SM < 0:
            soil[t] = 0
            SM = 0
        soil[t] = SM
        R[t] = rQ
        w[t] = inc

        # Groundwater storage
        SUZ += rQ
        if SUZ - PERC < 0:
            SLZ += SUZ
            SUZ = 0
        else:
            SLZ += PERC
            SUZ -= PERC

        if SUZ < UZL:
            Q_STZ[t] = 0
        else:
            Q_STZ[t] = (SUZ - UZL) * k0

        Q_SUZ[t] = SUZ * k1
        Q_SLZ[t] = SLZ * k2
        SUZ -= Q_SUZ[t] + Q_STZ[t]
        SLZ -= Q_SLZ[t]
        S1[t] = SUZ
        S2[t] = SLZ

        # Discharge components
        Qgen[t] = Q_STZ[t] + Q_SUZ[t] + Q_SLZ[t]

        # Total storage
        Storage[t] = S1[t] + S2[t] + soil[t]

    # Routing
    if routing == 1:
        step = 0.005
        i = np.arange(0, MAXBAS, step)
        h = np.zeros(len(i))

        j1 = i < MAXBAS / 2
        h[j1] = step * (i[j1] * 4 / MAXBAS ** 2)

        j2 = ~j1
        h[j2] = step * (4 / MAXBAS - i[j2] * 4 / MAXBAS ** 2)

        if MAXBAS % 1 > 0:
            I = np.linspace(1, len(i), int(MAXBAS + 1), dtype=int)
        else:
            I = np.linspace(1, len(i), int(MAXBAS), dtype=int)

        MAXBAS_w = np.array([np.sum(h[int(I[k - 1]):int(I[k])]) for k in range(1, len(I))])
        MAXBAS_w /= np.sum(MAXBAS_w)

        qs = np.convolve(Q_STZ, MAXBAS_w, mode='full')[:n_days]
        qi = np.convolve(Q_SUZ, MAXBAS_w, mode='full')[:n_days]
        qb = np.convolve(Q_SLZ, MAXBAS_w, mode='full')[:n_days]
        q = np.convolve(Qgen, MAXBAS_w, mode='full')[:n_days]
    else:
        qs = Q_STZ
        qi = Q_SUZ
        qb = Q_SLZ
        q = Qgen

    # Consolidate outputs into a dictionary
    out_dict = {
        # "q": q,
        "flow_sim": q,
        "qs": qs,
        "qi": qi,
        "qb": qb,
        "Storage": Storage,
        "SWE": SWE,
        "AET": AET,
        "SF": SF,
        "S1": S1,
        "S2": S2,
        "soil": soil,
        "w": w
    }

    return out_dict


def load_HBV_param(infile_param):
    # read HBV parameters from a pickle file

    try:
        df = pd.read_pickle(infile_param)
    except:
        df = pd.read_csv(infile_param)
    
    param1 = ['FC', 'beta', 'LP', 'SFCF', 'TT', 'CFMAX', 'k0', 'k1', 'k2', 'UZL', 'PERC']
    param2 = ['MAXBAS']
    
    param1_dict = {}
    for p in param1:
        param1_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    param2_dict = {}
    for p in param2:
        param2_dict[p] = df.loc[df['Parameter']==p]['Value'].iloc[0]

    return param1_dict, param2_dict


def save_HBV_output(filename, data, overwrite=True):
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        
def load_HBV_output(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def run_HBV_OneBasin_OneParamSet(P, T, PET, file_paramset, outfile, saveout = True):
    # run HBV on one parameter set

    if os.path.isfile(outfile):
        print('HBV output file exists:', outfile)
        print('Skip HBV modeling')

    else:
        # print('Run HBV using cold initial states')

        HBVparam_dict, routingparam_dict = load_HBV_param(file_paramset)
        HBV_simu = HBV_KM(P, T, PET, HBVparam_dict, routingparam_dict)

        if saveout:
            print('Saving output file to', outfile)
            save_HBV_output(outfile, HBV_simu)
