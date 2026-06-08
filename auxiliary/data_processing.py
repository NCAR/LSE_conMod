import pickle, toml, json
import pandas as pd
import numpy as np
from datetime import datetime

############
# forcing / flow / basin attribute data

def load_Yalan_selected_camels_id(inpath, IDfile=''):
    if len(IDfile)==0:
        subsetPath = f'{inpath}/Sub531ID.txt'
    else:
        subsetPath = f'{inpath}/{IDfile}'
    
    with open(subsetPath, 'r') as fp:
         selected_camels = json.load(fp)
    return selected_camels
    

def load_Yalan_data(inpath, startDate=None, endDate=None, IDfile=''):
    # data and codes shared by Yalan Song tested on the HBV model
    # deal with water year
    
    # inpath = '/glade/campaign/cgd/tss/people/guoqiang/HBV_CAMELS/data_Yalan'
    
    with open(f'{inpath}/training_file', 'rb') as f:
        forcing_train, target_train, attr_train = pickle.load(f)
    
    AllTime = pd.date_range('1980-10-01', f'2014-09-30', freq='d')
    
    gage_camels = np.load(f"{inpath}/gageid.npy") # all camels basins 671

    if len(IDfile)==0:
        subsetPath = f'{inpath}/Sub531ID.txt'
    else:
        subsetPath = f'{inpath}/{IDfile}'
    
    with open(subsetPath, 'r') as fp:
         selected_camels = json.load(fp)

    ### This method leads to mismatch between forcing_train/target_train and selected_camels!
    # [C, Ind, SubInd] = np.intersect1d(selected_camels, gage_camels, return_indices=True)

    # this method ensures correct mapping
    SubInd = []
    for i in selected_camels:
        SubInd.append(np.squeeze(np.where(gage_camels==i)))
    
    # startDate =1999
    # endDate = 2008
    if startDate == None:
        startDate = '1980-10-01'
    if endDate == None:
        endDate = '2014-09-30'
    newTime = pd.date_range(startDate, endDate, freq='d')
    
    index_start = AllTime.get_loc(newTime[0])
    index_end = AllTime.get_loc(newTime[-1]) + 1
    
    forcing_train = forcing_train[SubInd,index_start:index_end]
    
    target_train = target_train[SubInd,index_start:index_end]
    attr_train = attr_train[SubInd,:]
    
    attrLst = [ 'p_mean','pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
                'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
    
    forcingLst = ['prcp','tmean','pet']

    # convert flow unit from ft3/s to mm/day
    indarea = attrLst.index('area_gages2')
    for i in range(target_train.shape[0]):
        flowi = target_train[i, :, :]
        areai = attr_train[i, indarea]
        target_train[i, :, :] = flow_unit_conversion(flowi, areai, mode='cfs_to_mmhr')
    
    return forcing_train, target_train, attr_train, newTime, forcingLst, selected_camels, attrLst


def flow_unit_conversion(flow_data, basin_area=None, mode='cfs_to_mmd'):
    # mode == 'cfs_to_mmd': convert flow_data from ft³/s to mm/day
    # mode == 'mmd_to_cfs': convert flow_data from mm/day to ft³/s
    # mode == 'cfs_to_mhr': convert flow_data from ft³/s to mm/h 
    # mode == 'mhr_to_cfs': convert flow_data from mm/h (depth over basin) to ft³/s
    # basin_area unit: km²

    if mode == 'cfs_to_mmd':
        return (flow_data * 0.0283168 * 3600 * 24) / (basin_area * 1e6) * 1e3
    elif mode == 'mmd_to_cfs':
        return flow_data * (basin_area * 1e6 * 1e-3) / (0.0283168 * 3600 * 24)
    elif mode == 'cfs_to_mmhr':
        return (flow_data * 0.0283168 * 3600) / (basin_area * 1e6) * 1e3
    elif mode == 'mmhr_to_cfs':
        return flow_data * (basin_area * 1e6 * 1e-3) / (0.0283168 * 3600)
    else:
        raise ValueError("Invalid mode. Choose from 'cfs_to_mmd', 'mmd_to_cfs', 'cfs_to_mmhr', or 'mmhr_to_cfs'.")



# forcing / flow / basin attribute data
############

def get_index_of_nested_period(period_parent, period_subset):
    # example    
    # period_parent = ['1998-10-01', '2008-09-30']
    # period_subset = ['1999-10-01', '2008-09-30']
    
    # Convert the dates to datetime objects
    parent_start = datetime.strptime(period_parent[0], "%Y-%m-%d")
    parent_end = datetime.strptime(period_parent[1], "%Y-%m-%d")
    subset_start = datetime.strptime(period_subset[0], "%Y-%m-%d")
    subset_end = datetime.strptime(period_subset[1], "%Y-%m-%d")
    
    # Calculate the indexStart and indexEnd
    indexStart = (subset_start - parent_start).days
    indexEnd = (subset_end - parent_start).days

    return indexStart, indexEnd

def get_train_period(df, startDate, endDate, date_col=None):
    '''
    Generates the indices for the time period given in the configuration file.
    Differs per basin, so this function gets called before running the CFE. 
    
    Parameters:
        df: Forcing dataframe
        startDate: The beginning of the period
        endDate: The end of the period
        date_col (Optional): Name of your date column
    Returns:
        spinup_index, start_index, end_index: Appropriate indices for the df
    '''
    df = df.copy()

    # Input dates
    start = pd.Timestamp(startDate).normalize()
    end = pd.Timestamp(endDate).normalize().replace(hour=23)

    # ----------------------------------------
    # Spin-up starts at the provided startDate
    # ----------------------------------------
    spinup_mask = df[date_col] >= start

    # ----------------------------------------
    # Training starts at next water year
    # Water year starts Oct 1
    # ----------------------------------------
    if start.month >= 10:
        train_start = pd.Timestamp(year=start.year + 1, month=10, day=1)
    else:
        train_start = pd.Timestamp(year=start.year, month=10, day=1)

    train_mask = (
        (df[date_col] >= train_start) &
        (df[date_col] <= end)
    )

    # Indices
    spinup_index = df.index[spinup_mask].min()
    start_index = df.index[train_mask].min()
    end_index = df.index[train_mask].max()

    return spinup_index, start_index, end_index

def get_water_year_indices(df, n_years, spinup_years=1, date_col=None):
    """
    Grabs n_years from the df with the given number of spinup years for evaluation.
    
    ***Currently, the CFE has a specific time period for evaluation. So this function
    is not needed anymore, but may still be useful in the future***
    
    Parameters:
        df: Forcing dataframe
        n_years: The number of evaluation years to grab
        spinup_years (Optional): The number of years to count as spin-up
        date_col (Optional): Name of your date column
    Returns:
        spinup_idx, start_idx, end_idx,
        spinup_start, start_date, end_date

    Water year: Oct 1 00:00 → Sept 30 23:00
    """

    # Get datetime series
    if date_col:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = pd.to_datetime(df.index)
    
    first_date = dates.min()
    last_date  = dates.max()

    # Candidate water year: first possible Oct 1 after first_date
    candidate_year = first_date.year
    if first_date.month >= 10:
        candidate_year += 1

    while True:
        start_date = pd.Timestamp(candidate_year, 10, 1, 0)
        spinup_start = start_date - pd.DateOffset(years=spinup_years)

        # Ensure we have enough spinup and full eval window
        if spinup_start >= first_date:
            end_date = pd.Timestamp(candidate_year + n_years, 9, 30, 23)
            if end_date <= last_date:
                break

        candidate_year += 1

        if candidate_year > last_date.year:
            raise ValueError("Not enough data for spinup + evaluation years.")

    # --- indices ---
    spinup_idx = dates.searchsorted(spinup_start)
    start_idx  = dates.searchsorted(start_date)
    end_idx    = dates.searchsorted(end_date)

    return spinup_idx, start_idx, end_idx, spinup_start, start_date, end_date


def save_pickle_file(filename, data, overwrite=True):
    if ( not os.path.isfile(filename) ) or ( overwrite==True ):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        

def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


