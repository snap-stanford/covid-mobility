from covid_constants_and_util import *
from disease_model import Model
import matplotlib.ticker as ticker
from matplotlib import cm
import helper_methods_for_aggregate_data_analysis as helper
import seaborn as sns
import copy
from collections import Counter
import pickle
import re
import sys
import getpass
from traceback import print_exc
import socket
import psutil
import json
import subprocess
import multiprocessing
import IPython
import geopandas as gpd
from scipy.stats import scoreatpercentile
from psutil._common import bytes2human
from scipy.stats import ttest_ind, rankdata
from scipy.sparse import hstack
import argparse

PATH_TO_5_YEAR_ACS_DATA = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/2017_five_year_acs_data/2017_five_year_acs_data.csv'
TRAIN_TEST_PARTITION = datetime.datetime(2020, 4, 15)
PATH_TO_IPF_OUTPUT = '/dfs/scratch1/safegraph_homes/all_aggregate_data/ipf_output/'
AREA_CLIPPING_BELOW = 1
AREA_CLIPPING_ABOVE = 99
DWELL_TIME_CLIPPING_ABOVE = 90
HOURLY_VISITS_CLIPPING_ABOVE = 99

MSAS_TO_PRETTY_NAMES = {'Atlanta_Sandy_Springs_Roswell_GA':'Atlanta',
                        'Chicago_Naperville_Elgin_IL_IN_WI':"Chicago",
                        'Dallas_Fort_Worth_Arlington_TX':"Dallas",
                        'Houston_The_Woodlands_Sugar_Land_TX':"Houston",
                        'Los_Angeles_Long_Beach_Anaheim_CA':"Los Angeles",
                        'Miami_Fort_Lauderdale_West_Palm_Beach_FL':"Miami",
                        'New_York_Newark_Jersey_City_NY_NJ_PA':"New York City",
                        'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD':"Philadelphia",
                        'San_Francisco_Oakland_Hayward_CA':"San Francisco",
                        'Washington_Arlington_Alexandria_DC_VA_MD_WV':"Washington DC"}

MSAS_TO_STATE_CBG_FILES = {'Washington_Arlington_Alexandria_DC_VA_MD_WV':['ACS_2017_5YR_BG_11_DISTRICT_OF_COLUMBIA.gdb',
                                                        'ACS_2017_5YR_BG_24_MARYLAND.gdb',
                                                        'ACS_2017_5YR_BG_51_VIRGINIA.gdb',
                                                        'ACS_2017_5YR_BG_54_WEST_VIRGINIA.gdb'],
                      'Atlanta_Sandy_Springs_Roswell_GA':['ACS_2017_5YR_BG_13_GEORGIA.gdb'],
                      'Chicago_Naperville_Elgin_IL_IN_WI':['ACS_2017_5YR_BG_17_ILLINOIS.gdb',
                                                          'ACS_2017_5YR_BG_18_INDIANA.gdb',
                                                          'ACS_2017_5YR_BG_55_WISCONSIN.gdb'],
                      'Dallas_Fort_Worth_Arlington_TX':['ACS_2017_5YR_BG_48_TEXAS.gdb'],
                      'Houston_The_Woodlands_Sugar_Land_TX':['ACS_2017_5YR_BG_48_TEXAS.gdb'],
                      'Los_Angeles_Long_Beach_Anaheim_CA':['ACS_2017_5YR_BG_06_CALIFORNIA.gdb'],
                      'Miami_Fort_Lauderdale_West_Palm_Beach_FL':['ACS_2017_5YR_BG_12_FLORIDA.gdb'],
                      'New_York_Newark_Jersey_City_NY_NJ_PA':['ACS_2017_5YR_BG_36_NEW_YORK.gdb',
                                                              'ACS_2017_5YR_BG_34_NEW_JERSEY.gdb',
                                                              'ACS_2017_5YR_BG_42_PENNSYLVANIA.gdb'],
                      'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD':['ACS_2017_5YR_BG_42_PENNSYLVANIA.gdb',
                      'ACS_2017_5YR_BG_34_NEW_JERSEY.gdb',
                      'ACS_2017_5YR_BG_24_MARYLAND.gdb',
                      'ACS_2017_5YR_BG_10_DELAWARE.gdb'],
                      'San_Francisco_Oakland_Hayward_CA':['ACS_2017_5YR_BG_06_CALIFORNIA.gdb']} 

# in analysis, we remove same categories as MIT sloan paper, or try to. They write:
# We omit “Bars and Clubs” as SafeGraph seems to dramatically undercount these locations. We omit “Parks and Playgrounds” as SafeGraph struggles to precisely define the bor- ders of these irregularly shaped points of interest. We omit “Public and Private Schools” and “Child Care and Daycare Centers” due to challenges in adjusting for the fact that individuals under the age of 13 are not well tracked by SafeGraph.
SUBCATEGORY_BLACKLIST = ['Child Day Care Services',
'Elementary and Secondary Schools',
'Drinking Places (Alcoholic Beverages)',
'Nature Parks and Other Similar Institutions',
'General Medical and Surgical Hospitals',
'Other Airport Operations']

SUBCATEGORIES_TO_PRETTY_NAMES = {
    'Golf Courses and Country Clubs':'Golf Courses & Country Clubs',
    'Other Gasoline Stations':'Other Gas Stations',
    'Malls':'Malls',
    'Gasoline Stations with Convenience Stores':'Gas Stations',
    'New Car Dealers':'New Car Dealers',
    'Pharmacies and Drug Stores':'Pharmacies & Drug Stores',
    'Department Stores':'Department Stores',
    'Convenience Stores':'Convenience Stores',
    'All Other General Merchandise Stores':'Other General Stores',
    'Nature Parks and Other Similar Institutions':'Parks & Similar Institutions',
    'Automotive Parts and Accessories Stores':'Automotive Parts Stores',
    'Supermarkets and Other Grocery (except Convenience) Stores':'Grocery Stores',
    'Pet and Pet Supplies Stores':'Pet Stores',
    'Used Merchandise Stores':'Used Merchandise Stores',
    'Sporting Goods Stores':'Sporting Goods Stores',
    'Beer, Wine, and Liquor Stores':'Liquor Stores',
    'Insurance Agencies and Brokerages':'Insurance Agencies',
    'Gift, Novelty, and Souvenir Stores':'Gift Stores',
    'General Automotive Repair':'Car Repair Shops',
    'Limited-Service Restaurants':'Limited-Service Restaurants',
    'Snack and Nonalcoholic Beverage Bars':'Cafes & Snack Bars',
    'Offices of Physicians (except Mental Health Specialists)':'Offices of Physicians',
    'Fitness and Recreational Sports Centers':'Fitness Centers',
    'Musical Instrument and Supplies Stores':'Musical Instrument Stores',
    'Full-Service Restaurants':'Full-Service Restaurants',
    'Insurance Agencies':'Insurance Agencies',
    'Hotels (except Casino Hotels) and Motels':'Hotels & Motels',
    'Hardware Stores':'Hardware Stores',
    'Religious Organizations':'Religious Organizations',
    'Offices of Dentists':'Offices of Dentists',
    'Home Health Care Services':'Home Health Care Services',
    'Used Merchandise Stores':'Used Merchandise Stores',
    'General Medical and Surgical Hospitals':'General Hospitals',
    'Colleges, Universities, and Professional Schools':'Colleges & Universities',
    'Commercial Banking':'Commercial Banking',
    'Used Car Dealers':'Used Car Dealers',
    'Hobby, Toy, and Game Stores':'Hobby & Toy Stores',
    'Other Airport Operations':'Other Airport Operations',
    'Optical Goods Stores':'Optical Goods Stores',
    'Electronics Stores':'Electronics Stores',
    'Tobacco Stores':'Tobacco Stores',
    'All Other Amusement and Recreation Industries':'Other Recreation Industries',
    'Book Stores':'Book Stores',
    'Office Supplies and Stationery Stores':'Office Supplies',
    'Drinking Places (Alcoholic Beverages)':'Bars (Alc. Beverages)',
    'Furniture Stores':'Furniture Stores',
    'Assisted Living Facilities for the Elderly':'Senior Homes',
    'Sewing, Needlework, and Piece Goods Stores':'Sewing & Piece Goods Stores',
    'Cosmetics, Beauty Supplies, and Perfume Stores':'Cosmetics & Beauty Stores',
    'Amusement and Theme Parks':'Amusement & Theme Parks',
    'All Other Home Furnishings Stores':'Other Home Furnishings Stores',
    'Offices of Mental Health Practitioners (except Physicians)':'Offices of Mental Health Practitioners',
    'Carpet and Upholstery Cleaning Services':'Carpet Cleaning Services',
    'Florists':'Florists',
    'Women\'s Clothing Stores':'Women\'s Clothing Stores',
    'Family Clothing Stores':'Family Clothing Stores',
    'Jewelry Stores':'Jewelry Stores',
    'Beauty Salons':'Beauty Salons',
    'Motion Picture Theaters (except Drive-Ins)':'Movie Theaters',
    'Libraries and Archives':'Libraries & Archives',
    'Bowling Centers':'Bowling Centers',
    'Casinos (except Casino Hotels)':'Casinos',
}

MAX_MODELS_TO_TAKE_PER_MSA = 100 # previously 10,
ACCEPTABLE_LOSS_TOLERANCE = 1.2 # previously 1.5

###################################################
# Helper functions
###################################################
def get_cumulative(x):
    '''
    Converts an array of values into its cumulative form,
    i.e. cumulative_x[i] = x[0] + x[1] + ... + x[i]

    x should either be a 1D or 2D numpy array.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        cumulative_x = []
        curr_sum = 0
        for val in x:
            curr_sum = curr_sum + val
            cumulative_x.append(curr_sum)
        cumulative_x = np.array(cumulative_x)
    else:
        num_seeds, num_time = x.shape
        cumulative_x = []
        curr_sum = np.zeros(num_seeds)
        for i in range(num_time):
            curr_sum = curr_sum + x[:, i]
            cumulative_x.append(curr_sum.copy())
        cumulative_x = np.array(cumulative_x).T
    return cumulative_x

def get_daily_from_cumulative(x):
    '''
    Converts an array of values from its cumulative form
    back into its original form.

    x should either be a 1D or 2D numpy array.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        arr_to_return = np.array([x[0]] + list(x[1:] - x[:-1]))
    else:
        # seeds are axis 0, so want to subtract along axis 1.
        x0 = x[:, :1]
        increments = x[:, 1:] - x[:, :-1]
        arr_to_return = np.concatenate((x0, increments), axis=1)
    if not (arr_to_return >= 0).all():
        bad_val_frac = (arr_to_return < 0).mean()
        print("Warning: fraction %2.3f of values are not greater than 0! clipping to 0" % bad_val_frac)
        print(arr_to_return)
        assert bad_val_frac < 0.1 # this happens quite occasionally in NYT data.
        arr_to_return = np.clip(arr_to_return, 0, None)
    return arr_to_return

def MRE(y_true, y_pred):
    '''
    Computes the median relative error (MRE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the MRE is returned.
    If y_true and y_pred are 2D, e.g. predictions over multiple seeds,
    then the mean of the MREs is returned.
    '''
    abs_err = np.absolute(y_true - y_pred)
    rel_err = abs_err / y_true
    if len(abs_err.shape) == 1:
        mre = np.median(rel_err)
    else:
        mre = np.mean(np.median(rel_err, axis=1))
    return mre

def RMSE(y_true, y_pred, agg_over_dimensions=True):
    '''
    Computes the root mean squared error (RMSE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the RMSE is returned.
    If y_true and y_pred are 2D, e.g. predictions over multiple seeds,
    then the mean of the RMSEs is returned.
    '''
    sq_err = (y_true - y_pred) ** 2
    if len(sq_err.shape) == 1:  # this implies y_true and y_pred are 1D
        rmse = np.sqrt(np.mean(sq_err))
    else:  # this implies at least one of them is 2D
        rmse = np.sqrt(np.mean(sq_err, axis=1))
        if agg_over_dimensions:
            rmse = np.mean(rmse)
    return rmse

def gaussianish_negative_ll(y_true, y_pred):
    # y_pred should be n_seeds x n_timesteps
    assert len(y_pred.shape) == 2
    assert len(y_true.shape) == 1
    sq_err = (y_true - y_pred) ** 2
    variance_across_seeds = np.std(y_pred, axis=0, ddof=1) ** 2
    ll = np.sum(-.5 * np.log(variance_across_seeds) - .5 * sq_err/variance_across_seeds)
    return -ll

def MSE(y_true, y_pred):
    '''
    Computes the mean squared error (MSE). y_true and y_pred should
    both be numpy arrays.
    '''
    return np.mean((y_true - y_pred) ** 2)

def mean_and_CIs_of_timeseries_matrix(M, alpha=0.05):
    """
    Given a matrix which is N_SEEDS X T, return mean and upper and lower CI for plotting.
    """
    assert alpha > 0
    assert alpha < 1
    mean = np.mean(M, axis=0)
    lower_CI = np.percentile(M, 100 * alpha/2, axis=0)
    upper_CI = np.percentile(M, 100 * (1 - alpha/2), axis=0)
    return mean, lower_CI, upper_CI

def get_datetime_hour_as_string(datetime_hour):
    return '%i.%i.%i.%i' % (datetime_hour.year, datetime_hour.month,
                            datetime_hour.day, datetime_hour.hour)

def apply_smoothing(x, before=2, after=2):
    new_x = []
    for i, x_point in enumerate(x):
        before_idx = max(0, i-before)
        after_idx = min(len(x), i+after+1)
        new_x.append(np.mean(x[before_idx:after_idx]))
    return np.array(new_x)

###################################################
# Code for running experiments
###################################################

def fit_disease_model_on_real_data(d,
                                   min_datetime,
                                   max_datetime,
                                   exogenous_model_kwargs,
                                   poi_attributes_to_clip,
                                   correct_poi_visits=True,
                                   aggregate_col_to_use='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                   cbg_count_cutoff=10,
                                   cbgs_to_seed_in=None,
                                   model_init_kwargs=None,
                                   simulation_kwargs=None,
                                   multiply_poi_visit_counts_by_census_ratio=True,
                                   cbg_groups_to_track=None,
                                   counties_to_track=None,
                                   only_track_attributes_of_cbgs_in_msa=False,
                                   include_cbg_prop_out=False,
                                   filter_for_cbgs_in_msa=False,
                                   cbgs_to_filter_for=None,
                                   cbg_day_prop_out=None,
                                   return_model_without_fitting=False,
                                   return_model_and_data_without_fitting=False,
                                   use_density_based_home_rates=False,
                                   include_poi_dwell_time_correction=True,
                                   counterfactual_poi_opening_experiment_kwargs=None,
                                   counterfactual_retrospective_experiment_kwargs=None,
                                   preload_poi_visits_list_filename=None,
                                   poi_cbg_visits_list=None,
                                   verbose=True):
    """
    Function to format real data as input for the disease model, and to run the
    disease simulation on the data.
    d: DataFrame of POIs
    min_datetime, max_datetime: the first and last hour to pull from d hourly_visits
    aggregate_col_to_use: the field that holds the aggregated CBG proportions for each POI
    cbg_count_cutoff: the minimum number of POIs in the data that a CBG must visit
                      to be included in the model
    cbgs_to_seed_in: list of CBG names; specifies which CBGs start off with sick people
    model_init_kwargs: extra arguments for initializing Model
    exogenous_model_kwargs: extra arguments for Model.init_exogenous_variables()
    simulation_kwargs: extra arguments for Model.simulate_disease_spread()
    multiply_poi_visit_counts_by_census_ratio: if True, upscale visit counts by a constant factor
                                               derived using Census data to try to get real visit
                                               volumes
    cbg_groups_to_track: dict of group to CBG names; the CBGs to keep track of during simulation
    include_cbg_prop_out: whether to use IPF to adjust the POI-CBG proportions based on Social Distancing Metrics (SDM)
    cbg_day_prop_out: a DataFrame of CBGs and their SDM-based proportions out per day

    -------------------------
    Sample usage:

    d = helper.load_dataframe_for_individual_msa('San_Francisco_Oakland_Hayward_CA')
    fit_disease_model_on_real_data(d,
                                   min_datetime=datetime.datetime(2020, 3, 1, 0))
    -------------------------

    """
    if model_init_kwargs is None:
        model_init_kwargs = {}
    if simulation_kwargs is None:
        simulation_kwargs = {}
    if cbg_groups_to_track is None:
        cbg_groups_to_track = {}
    assert 'p_sick_at_t0' in exogenous_model_kwargs  # required for model initialization, has no default value
    assert aggregate_col_to_use in ['aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                    'aggregated_visitor_home_cbgs']
    if use_density_based_home_rates:
        assert 'home_psi' in exogenous_model_kwargs
    assert not (return_model_without_fitting and return_model_and_data_without_fitting)

    if preload_poi_visits_list_filename is not None:
        f = open(preload_poi_visits_list_filename, 'rb')
        poi_cbg_visits_list = pickle.load(f)
        f.close()
    if poi_cbg_visits_list is not None:
        # can't provide both cbg_prop_out (which triggers IPF) and precomputed poi_cbg matrices
        assert not include_cbg_prop_out

    t0 = time.time()
    print('1. Processing SafeGraph data...')
    # get hour column strings.
    assert min_datetime <= max_datetime
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    if poi_cbg_visits_list is not None:
        assert len(poi_cbg_visits_list) == len(all_hours)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    assert(all([col in d.columns for col in hour_cols]))
    print("Found %d hours in all (%s to %s) -> %d hourly visits" % (len(all_hours),
         get_datetime_hour_as_string(min_datetime),
         get_datetime_hour_as_string(max_datetime),
         np.nansum(d[hour_cols].values)))
    all_states = sorted(list(set(d['region'].dropna())))
    
    # aggregate median_dwell time over weeks.
    weekly_median_dwell_pattern = re.compile('2020-\d\d-\d\d.median_dwell')
    median_dwell_cols = [col for col in d.columns if re.match(weekly_median_dwell_pattern, col)]
    print('Aggregating median_dwell from %s to %s' % (median_dwell_cols[0], median_dwell_cols[-1]))
    # note: this may trigger "RuntimeWarning: All-NaN slice encountered" if a POI has all nans for median_dwell; 
    # this is not a problem and will be addressed in apply_percentile_based_clipping_to_msa_df
    avg_dwell_times = d[median_dwell_cols].median(axis=1).values
    nan_idx = np.isnan(avg_dwell_times)
    print('%d/%d POIs with all nans in median_dwell columns -> will fill in with median dwell time of category' % (np.sum(nan_idx), len(d)))
    d['avg_median_dwell'] = avg_dwell_times

    # clip before dropping data so we have more POIs as basis for percentiles.
    assert all([key in poi_attributes_to_clip for key in ['clip_areas', 'clip_dwell_times', 'clip_visits']])
    d, categories_to_clip, cols_to_clip, thresholds, medians = impute_missing_values_and_clip_poi_attributes(
        d, min_datetime, max_datetime, inplace=True, min_pois_in_cat=100, **poi_attributes_to_clip)
    print('Finished clipping -> %d hourly visits' % np.nansum(d[hour_cols].values))
    
    # remove POIs with missing data.
    if verbose: print("Prior to dropping any POIs, %i POIs" % len(d))
    d = d.dropna(subset=hour_cols)
    if verbose: print("After dropping for missing hourly visits, %i hourly visits, %i POIs" % (d[hour_cols].values.sum(), len(d)))
    d = d.loc[d[aggregate_col_to_use].map(lambda x:len(x.keys()) > 0)]
    if verbose: print("After dropping for missing CBG home data, %i hourly visits, %i POIs" % (d[hour_cols].values.sum(), len(d)))
    d = d.dropna(subset=['avg_median_dwell'])
    # this is a special case where avg_median_dwell was not able to be filled in based on other POIs in the same category
    if verbose: print("After dropping for missing avg_median_dwell, %i hourly visits, %i POIs" % (d[hour_cols].values.sum(), len(d)))   

    # reindex CBGs.
    poi_cbg_proportions = d[aggregate_col_to_use].values
    all_cbgs = [a for b in poi_cbg_proportions for a in b.keys()]
    cbg_counts = Counter(all_cbgs).most_common()
    # only keep CBGs that have visited at least this many POIs
    all_unique_cbgs = [cbg for cbg, count in cbg_counts if count >= cbg_count_cutoff]
    # if filtering for CBGs in MSA, only keep CBGs in MSA.
    if filter_for_cbgs_in_msa:
        assert cbgs_to_filter_for is not None
        print("Prior to filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
        all_unique_cbgs = [a for a in all_unique_cbgs if a in cbgs_to_filter_for]
        print("After filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
    else:
        assert cbgs_to_filter_for is None
    # order CBGs lexicographically
    all_unique_cbgs = sorted(all_unique_cbgs)
    N = len(all_unique_cbgs)
    if verbose: print("After dropping CBGs that appear in < %i POIs, %i CBGs (%2.1f%%)" %
          (cbg_count_cutoff, N, 100.*N/len(cbg_counts)))
    cbgs_to_idxs = dict(zip(all_unique_cbgs, range(N)))

    # convert data structures with CBG names to CBG indices
    poi_cbg_proportions_int_keys = []
    kept_poi_idxs = []
    E = 0  # number of connected POI-CBG pairs (ignoring time)
    for poi_idx, old_dict in enumerate(poi_cbg_proportions):
        new_dict = {}
        for string_key in old_dict:
            if string_key in cbgs_to_idxs:
                int_key = cbgs_to_idxs[string_key]
                new_dict[int_key] = old_dict[string_key]
                E += 1
        if len(new_dict) > 0:
            poi_cbg_proportions_int_keys.append(new_dict)
            kept_poi_idxs.append(poi_idx)
    M = len(kept_poi_idxs)
    if verbose:
        print('Dropped %d POIs whose visitors all come from dropped CBGs' %
              (len(poi_cbg_proportions) - M))
    print('FINAL: number of CBGs (N) = %d, number of POIs (M) = %d' % (N, M))
    print('Num connected POI-CBG pairs (E) = %d, network density (E/N) = %.3f' %
          (E, E / N))  # avg num POIs per CBG
    
    # get POI-related variables
    d = d.iloc[kept_poi_idxs]
    poi_subcategory_types = d['sub_category'].values
    poi_areas = d['safegraph_computed_area_in_square_feet'].values
    poi_dwell_times = d['avg_median_dwell'].values
    if include_poi_dwell_time_correction:
        poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
        print('Dwell time correction factors: mean = %.2f, min = %.2f, max = %.2f' %
              (np.mean(poi_dwell_time_correction_factors), min(poi_dwell_time_correction_factors), max(poi_dwell_time_correction_factors)))
    else:
        poi_dwell_time_correction_factors = None
    poi_time_counts = d[hour_cols].values
    if correct_poi_visits:
        print('Correcting POI hourly visit vectors...')
        new_poi_time_counts = []
        for i, (visit_vector, dwell_time) in enumerate(list(zip(poi_time_counts, poi_dwell_times))):
            new_poi_time_counts.append(correct_visit_vector(visit_vector, dwell_time))
        poi_time_counts = np.array(new_poi_time_counts)
        d = d.copy()
        d[hour_cols] = poi_time_counts
        new_hourly_visit_count = np.sum(poi_time_counts)
        print('After correcting, %.2f hourly visits' % new_hourly_visit_count)    

    # Print out most common POI categories
    pd.set_option("display.width", 500)
    poi_categories = d[['top_category', 'sub_category']].copy()
    poi_categories['total_visits'] = poi_time_counts.sum(axis=1)
    poi_category_counts = poi_categories.groupby(['top_category', 'sub_category']).agg(['size', 'sum']).reset_index()
    poi_category_counts.columns = ['top_category', 'sub_category', 'n_pois', 'n_visits']
    poi_category_counts = poi_category_counts.sort_values(by='n_visits')[::-1]
    poi_category_counts['prop_pois'] = poi_category_counts['n_pois'] / poi_category_counts['n_pois'].sum()
    poi_category_counts['prop_visits'] = poi_category_counts['n_visits'] / poi_category_counts['n_visits'].sum()
    # truncate strings for printing
    poi_category_counts['top_category_short'] = poi_category_counts['top_category'].map(lambda x:x[:30])
    poi_category_counts['sub_category_short'] = poi_category_counts['sub_category'].map(lambda x:x[:30])
    poi_category_counts['cum_prop_visits'] = poi_category_counts['prop_visits'].cumsum()
    print("\n\nMost common POI categories")
    print(poi_category_counts.head(n=30)[['top_category_short', 'sub_category_short', 'prop_visits', 'cum_prop_visits', 'prop_pois']])
    assert poi_category_counts['n_pois'].sum() / len(poi_areas) > .99
    
    cbg_idx_groups_to_track = {}
    for group in cbg_groups_to_track:
        cbg_idx_groups_to_track[group] = [
            cbgs_to_idxs[a] for a in cbg_groups_to_track[group] if a in cbgs_to_idxs]
        if verbose: print(f'{len(cbg_groups_to_track[group])} CBGs in {group} -> matched {len(cbg_idx_groups_to_track[group])} ({(len(cbg_idx_groups_to_track[group]) / len(cbg_groups_to_track[group])):.3f})')
    if cbgs_to_seed_in is None:
        cbg_idx_to_seed_in = None
    else:
        cbg_idx_to_seed_in = {cbgs_to_idxs[a] for a in cbgs_to_seed_in if a in cbgs_to_idxs}

    # get CBG-related variables from census data
    print('2. Processing ACS data...')
    acs_d = helper.load_and_reconcile_multiple_acs_data()
    # use most recent population data
    cbgs_to_census_pops = dict(zip(acs_d['census_block_group'].values,
                                   acs_d['total_cbg_population_2018_1YR'].values))
    cbg_sizes = np.array([cbgs_to_census_pops[a] for a in all_unique_cbgs])
    assert np.sum(np.isnan(cbg_sizes)) == 0
    if verbose:
        print('CBGs: median population size = %d, sum of population sizes = %d' %
          (np.median(cbg_sizes), np.sum(cbg_sizes)))

    if use_density_based_home_rates:
        cbgs_to_densities = dict(zip(acs_d['census_block_group'].values, acs_d['people_per_mile_hybrid'].values))
        cbg_densities = np.array([cbgs_to_densities[a] for a in all_unique_cbgs])
        nan_idx = np.isnan(cbg_densities)
        non_nan_vals = cbg_densities[~nan_idx]
        median_density = np.median(non_nan_vals)
        if verbose:
            print('%d CBGs missing densities, filling with median density of %.2f' %
              (np.sum(nan_idx), median_density))
        cbg_densities[nan_idx] = median_density
    else:
        cbg_densities = None  # if None, then model will use uniform home_beta

    if multiply_poi_visit_counts_by_census_ratio:
        # Get overall undersampling factor.
        # Basically we take ratio of ACS US population to SafeGraph population in Feb 2020.
        # SafeGraph seems to think this is reasonable.
        # https://safegraphcovid19.slack.com/archives/C0109NPA543/p1586801883190800?thread_ts=1585770817.335800&cid=C0109NPA543
        total_us_population_in_50_states_plus_dc = acs_d.loc[acs_d['state_code'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC), 'total_cbg_population_2018_1YR'].sum()
        safegraph_visitor_count_df = pd.read_csv('/dfs/scratch1/safegraph_homes/all_aggregate_data/20191213-safegraph-aggregate-longitudinal-data-to-unzip-to/SearchofAllRecords-CORE_POI-GEOMETRY-PATTERNS-2020_02-2020-03-16/visit_panel_summary.csv')
        safegraph_visitor_count = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['state'] == 'ALL_STATES', 'num_unique_visitors'].iloc[0]

        # remove a few safegraph visitors from non-US states.
        two_letter_codes_for_states = set([a.lower() for a in codes_to_states if codes_to_states[a] in JUST_50_STATES_PLUS_DC])
        safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['state'].map(lambda x:x not in two_letter_codes_for_states and x != 'ALL_STATES'), 'num_unique_visitors'].sum()
        if verbose:
            print("Removing %2.3f%% of people from SafeGraph count who are not in 50 states or DC" %
                (100. * safegraph_visitor_count_to_non_states/safegraph_visitor_count))
        safegraph_visitor_count = safegraph_visitor_count - safegraph_visitor_count_to_non_states
        correction_factor = 1. * total_us_population_in_50_states_plus_dc / safegraph_visitor_count
        if verbose: 
            print("Total US population from ACS: %i; total safegraph visitor count: %i; correction factor for POI visits is %2.3f" %
                (total_us_population_in_50_states_plus_dc,
                safegraph_visitor_count,
                correction_factor))
        poi_time_counts = poi_time_counts * correction_factor

    # only include CBGs from MSA in attribute tracking
    if only_track_attributes_of_cbgs_in_msa and 'nyt' in cbg_idx_groups_to_track:
        cbg_idx_to_track = set(cbg_idx_groups_to_track['nyt'])
        print('Only including CBGs from MSA in attribute tracking (%i CBGs)' % len(cbg_idx_to_track))
    else:
        cbg_idx_to_track = set(range(N))  # include all CBGs
    if counties_to_track is not None:
        print('Found %d counties to track...' % len(counties_to_track))
        county2cbgs = {}
        total_cbgs = 0
        cbgs_modeled = 0
        for county in counties_to_track:
            county_cbgs = acs_d[acs_d['county_code'] == county]['census_block_group'].values
            orig_len = len(county_cbgs)
            county_cbgs = sorted(set(county_cbgs).intersection(set(all_unique_cbgs)))
            #assert len(county_cbgs)/orig_len > .9 # if we are losing a lot of CBGs we need to do a better case count correction.
            total_cbgs += orig_len
            cbgs_modeled += len(county_cbgs)
            if len(counties_to_track) <= 20:
                print('County %i: found %d cbgs in county, keeping %d that are in model' % (county, orig_len, len(county_cbgs)))
            if len(county_cbgs) > 0:
                county_cbg_idx = np.array([cbgs_to_idxs[a] for a in county_cbgs])
                county2cbgs[county] = (county_cbgs, county_cbg_idx)
        print('Found cbgs in model for %d of the counties' % len(county2cbgs))
        print("%i/%i CBGs in counties were modeled (proportion %2.3f)" % (cbgs_modeled, total_cbgs, cbgs_modeled/total_cbgs))
        assert cbgs_modeled/total_cbgs > .9
    else:
        county2cbgs = None

    # turn off warnings temporarily so that using > or <= on np.nan does not cause warnings
    np.warnings.filterwarnings('ignore')
    for attribute in ['p_black', 'p_white', 'median_household_income']:
        attr_col_name = '%s_2017_5YR' % attribute  # using 5-year ACS data for attributes bc less noisy
        assert attr_col_name in acs_d.columns
        mapper_d = dict(zip(acs_d['census_block_group'].values, acs_d[attr_col_name].values))
        attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in all_unique_cbgs])
        non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
        median_cutoff = np.median(non_nan_vals)
        if verbose:
            print("Attribute %s: was able to compute for %2.1f%% out of %i CBGs, median is %2.3f" %
                (attribute, 100. * len(non_nan_vals) / len(cbg_idx_to_track),
                 len(cbg_idx_to_track), median_cutoff))

        cbg_idx_groups_to_track[f'{attribute}_above_median'] = list(set(np.where(attribute_vals > median_cutoff)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_below_median'] = list(set(np.where(attribute_vals <= median_cutoff)[0]).intersection(cbg_idx_to_track))

        top_decile = scoreatpercentile(non_nan_vals, 90)
        bottom_decile = scoreatpercentile(non_nan_vals, 10)
        cbg_idx_groups_to_track[f'{attribute}_top_decile'] = list(set(np.where(attribute_vals >= top_decile)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_bottom_decile'] = list(set(np.where(attribute_vals <= bottom_decile)[0]).intersection(cbg_idx_to_track))

        if counties_to_track is not None:
            above_median_in_county = []
            below_median_in_county = []
            for county in county2cbgs:
                county_cbgs, cbg_idx = county2cbgs[county]
                attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in county_cbgs])
                non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
                median_cutoff = np.median(non_nan_vals)
                above_median_idx = cbg_idx[np.where(attribute_vals > median_cutoff)[0]]
                above_median_idx = list(set(above_median_idx).intersection(cbg_idx_to_track))
                #cbg_idx_groups_to_track[f'{county}_{attribute}_above_median'] = above_median_idx
                above_median_in_county.extend(above_median_idx)
                below_median_idx = cbg_idx[np.where(attribute_vals <= median_cutoff)[0]]
                below_median_idx = list(set(below_median_idx).intersection(cbg_idx_to_track))
                #cbg_idx_groups_to_track[f'{county}_{attribute}_below_median'] = below_median_idx
                below_median_in_county.extend(below_median_idx)
            cbg_idx_groups_to_track[f'{attribute}_above_median_in_own_county'] = above_median_in_county
            cbg_idx_groups_to_track[f'{attribute}_below_median_in_own_county'] = below_median_in_county
    np.warnings.resetwarnings()

    if include_cbg_prop_out:
        model_days = helper.list_datetimes_in_range(min_datetime, max_datetime)
        cols_to_keep = ['%s.%s.%s' % (dt.year, dt.month, dt.day) for dt in model_days]
        print('Giving model prop out for %s to %s' % (cols_to_keep[0], cols_to_keep[-1]))
        assert((len(cols_to_keep) * 24) == len(hour_cols))
        if cbg_day_prop_out is None:
            print('Loading Social Distancing Metrics and computing per-day CBG prop out: warning, this could take a while...')
            sdm_mdl = helper.load_social_distancing_metrics(model_days)
            cbg_day_prop_out = helper.compute_cbg_day_prop_out(sdm_mdl, all_unique_cbgs)
        else:
            all_cbgs_set = set(all_unique_cbgs)
            cbg_day_prop_out = cbg_day_prop_out[cbg_day_prop_out['census_block_group'].isin(all_cbgs_set)]
            missing_cbgs = all_cbgs_set - set(cbg_day_prop_out.census_block_group)
            print('%d CBGs missing reweighting info -> filling with median values' % len(missing_cbgs))
            assert(all([col in cbg_day_prop_out.columns for col in cols_to_keep]))
            if len(missing_cbgs) > 0:
                all_prop_out = cbg_day_prop_out[cols_to_keep].values
                # fill in missing CBGs with median reweightings
                median_prop_out = np.median(all_prop_out, axis=0)
                missing_prop_out = np.broadcast_to(median_prop_out, (len(missing_cbgs), len(cols_to_keep)))
                missing_prop_out_df = pd.DataFrame(missing_prop_out, columns=cols_to_keep)
                missing_prop_out_df['census_block_group'] = list(missing_cbgs)
                cbg_day_prop_out = pd.concat((cbg_day_prop_out, missing_prop_out_df), sort=False)
        assert(len(cbg_day_prop_out) == len(all_unique_cbgs))
        # sort lexicographically, like all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out.sort_values(by='census_block_group')
        assert list(cbg_day_prop_out['census_block_group'].values) == all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out[cols_to_keep].values

    # If trying to get the counterfactual where social activity doesn't change, just repeat first week of dataset.
    # We put this in exogenous_model_kwargs because it actually affects how the model runs, not just the data input.
    if 'just_compute_r0' in exogenous_model_kwargs and exogenous_model_kwargs['just_compute_r0']:
        print('Running model to compute r0 -> looping first week visit counts')
        # simulate out 15 weeks just so we are sure all cases are gone.
        max_datetime = min_datetime + datetime.timedelta(hours=(168*15)-1)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        if poi_cbg_visits_list is not None:
            assert len(poi_cbg_visits_list) >= 168
            new_visits_list = []
            for i in range(168 * 15):
                first_week_idx = i % 168  # map to corresponding hour in first week
                new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
            poi_cbg_visits_list = new_visits_list
            assert len(poi_cbg_visits_list) == len(all_hours)
        else:
            assert poi_time_counts.shape[1] >= 168  # ensure that we have at least a week to model
            first_week = poi_time_counts[:, :168]
            poi_time_counts = np.tile(first_week, (1, 15))
            if cbg_day_prop_out is not None:
                assert cbg_day_prop_out.shape[1] >= 7
                first_week = cbg_day_prop_out[:, :7]
                cbg_day_prop_out = np.tile(first_week, (1, 15))
            assert poi_time_counts.shape[1] == len(all_hours)

    # if we want to run counterfactual reopening simulations
    intervention_cost = None
    if counterfactual_poi_opening_experiment_kwargs is not None:
        if poi_cbg_visits_list is None:
            raise Exception('Reopening experiments are only implemented when IPF output is provided')
        extra_weeks_to_simulate = counterfactual_poi_opening_experiment_kwargs['extra_weeks_to_simulate']
        assert extra_weeks_to_simulate >= 0
        new_all_hours = helper.list_hours_in_range(min_datetime, max_datetime + datetime.timedelta(hours=168 * extra_weeks_to_simulate))
        intervention_datetime = counterfactual_poi_opening_experiment_kwargs['intervention_datetime']
        assert(intervention_datetime in new_all_hours)
        intervention_hour_idx = new_all_hours.index(intervention_datetime)
        if 'top_category' in counterfactual_poi_opening_experiment_kwargs:
            top_category = counterfactual_poi_opening_experiment_kwargs['top_category']
        else:
            top_category = None
        if 'sub_category' in counterfactual_poi_opening_experiment_kwargs:
            sub_category = counterfactual_poi_opening_experiment_kwargs['sub_category']
        else:
            sub_category = None

        # must have one but not both of these arguments
        assert 'alpha' in counterfactual_poi_opening_experiment_kwargs or 'full_activity_alpha' in counterfactual_poi_opening_experiment_kwargs
        # the original alpha - post-intervention is interpolation between no reopening and full activity
        if 'alpha' in counterfactual_poi_opening_experiment_kwargs:
            assert 'full_activity_alpha' not in counterfactual_poi_opening_experiment_kwargs
            alpha = counterfactual_poi_opening_experiment_kwargs['alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list,
                                        poi_categories, poi_areas, new_all_hours, intervention_hour_idx,
                                        alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=True)
        # post-intervention is alpha-percent of full activity (no interpolation)
        else:
            assert 'alpha' not in counterfactual_poi_opening_experiment_kwargs
            alpha = counterfactual_poi_opening_experiment_kwargs['full_activity_alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list,
                                        poi_categories, poi_areas, new_all_hours, intervention_hour_idx,
                                        alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=False)

        # should be used in tandem with alpha or full_activity_alpha, since the timeseries is extended
        # in those blocks; this part just caps post-intervention visits to alpha-percent of max capacity
        if 'max_capacity_alpha' in counterfactual_poi_opening_experiment_kwargs:
            if poi_cbg_visits_list is None:
                raise Exception('Max capacity experiment is only implemented when IPF output is provided')
            max_capacity_alpha = counterfactual_poi_opening_experiment_kwargs['max_capacity_alpha']
            assert max_capacity_alpha >= 0 and max_capacity_alpha <= 1
            poi_visits = np.zeros((M, len(all_hours)))   # num pois x length of original data
            for t, poi_cbg_visits in enumerate(poi_cbg_visits_list[:len(all_hours)]):
                poi_visits[:, t] = poi_cbg_visits @ np.ones(N)
            max_per_poi = np.max(poi_visits, axis=1)
            alpha_max_per_poi = np.clip(max_capacity_alpha * max_per_poi, 1e-10, None)  # so that we don't divide by 0
            orig_total_activity = 0
            capped_total_activity = 0
            for t in range(intervention_hour_idx, len(poi_cbg_visits_list)):
                poi_cbg_visits = poi_cbg_visits_list[t]
                num_visits_per_poi = poi_cbg_visits @ np.ones(N)
                orig_total_activity += np.sum(num_visits_per_poi)
                ratio_per_poi = num_visits_per_poi / alpha_max_per_poi
                clipping_idx = ratio_per_poi > 1
                poi_multipliers = np.ones(M)
                poi_multipliers[clipping_idx] = 1 / ratio_per_poi[clipping_idx]
                adjusted_poi_cbg_visits = poi_cbg_visits.transpose().multiply(poi_multipliers).transpose().tocsr()
                capped_total_activity += np.sum(adjusted_poi_cbg_visits @ np.ones(N))
                poi_cbg_visits_list[t] = adjusted_poi_cbg_visits
            print('Finished capping visits at %.1f%% of max capacity -> kept %.4f%% of visits' %
                  (100. * max_capacity_alpha, 100 * capped_total_activity / orig_total_activity))
            intervention_cost['total_activity_after_max_capacity_capping'] = capped_total_activity
        all_hours = new_all_hours
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))

    if counterfactual_retrospective_experiment_kwargs is not None:
        # must have one but not both of the two arguments
        assert 'distancing_degree' in counterfactual_retrospective_experiment_kwargs or 'shift_in_days' in counterfactual_retrospective_experiment_kwargs
        if poi_cbg_visits_list is None:
            raise Exception('Retrospective experiments are only implemented for when poi_cbg_visits_list is precomputed')
        new_visits_list = []
        if 'distancing_degree' in counterfactual_retrospective_experiment_kwargs:
            assert 'shift_in_days' not in counterfactual_retrospective_experiment_kwargs
            distancing_degree = counterfactual_retrospective_experiment_kwargs['distancing_degree']
            for i, m in enumerate(poi_cbg_visits_list):
                if i < 168:  # first week
                    new_visits_list.append(m.copy())
                else:
                    first_week_m = poi_cbg_visits_list[i % 168]
                    mixture = first_week_m.multiply(1-distancing_degree) + m.multiply(distancing_degree)
                    new_visits_list.append(mixture.copy())
            print('Modified poi_cbg_visits_list for retrospective experiment: distancing_degree = %s.' % distancing_degree)
        else:
            assert 'distancing_degree' not in counterfactual_retrospective_experiment_kwargs
            shift_in_days = counterfactual_retrospective_experiment_kwargs['shift_in_days']
            shift_in_hours = shift_in_days * 24
            if shift_in_hours <= 0:  # shift earlier
                new_visits_list = [m.copy() for m in poi_cbg_visits_list[abs(shift_in_hours):]]
                current_length = len(new_visits_list)
                assert current_length >= 168
                last_week = new_visits_list[-168:]
                for i in range(current_length, len(poi_cbg_visits_list)):
                    last_week_counterpart = last_week[i % 168].copy()
                    new_visits_list.append(last_week_counterpart)
            else:  # shift later
                for i in range(len(poi_cbg_visits_list)):
                    if i-shift_in_hours < 0:
                        distance_from_start = (shift_in_hours - i) % 168
                        first_week_idx = 168 - distance_from_start
                        new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
                    else:
                        new_visits_list.append(poi_cbg_visits_list[i-shift_in_hours].copy())
            print('Modified poi_cbg_visits_list for retrospective experiment: shifted by %d days.' % shift_in_days)
        poi_cbg_visits_list = new_visits_list

    print('Total time to prep data: %.3fs' % (time.time() - t0))

    # feed everything into model.
    m = Model(**model_init_kwargs)
    m.init_exogenous_variables(poi_cbg_proportions=poi_cbg_proportions_int_keys,
                               poi_time_counts=poi_time_counts,
                               poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               cbg_densities=cbg_densities,
                               all_unique_cbgs=all_unique_cbgs,
                               cbgs_to_idxs=cbgs_to_idxs,
                               all_states=all_states,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
                               cbg_idx_to_seed_in=cbg_idx_to_seed_in,
                               cbg_idx_groups_to_track=cbg_idx_groups_to_track,
                               cbg_day_prop_out=cbg_day_prop_out,
                               intervention_cost=intervention_cost,
                               poi_subcategory_types=poi_subcategory_types,
                               **exogenous_model_kwargs)
    m.init_endogenous_variables()
    if return_model_without_fitting:
        return m
    elif return_model_and_data_without_fitting:
        m.d = d
        return m
    m.simulate_disease_spread(**simulation_kwargs)
    return m

def correct_visit_vector(v, median_dwell_in_minutes):
    """
    Given an original hourly visit vector v and a dwell time in minutes, 
    return a new hourly visit vector which accounts for spillover. 
    """
    v = np.array(v)
    d = median_dwell_in_minutes/60.
    new_v = v.copy().astype(float)
    max_shift = math.floor(d + 1) # maximum hours we can spill over to. 
    for i in range(1, max_shift + 1):
        if i < max_shift:
            new_v[i:] += v[:-i] # this hour is fully occupied
        else:
            new_v[i:] += (d - math.floor(d)) * v[:-i] # this hour only gets part of the visits. 
    return new_v

def impute_missing_values_and_clip_poi_attributes(d, min_datetime, max_datetime,
                                                  clip_areas, clip_dwell_times, clip_visits,
                                                  area_below=AREA_CLIPPING_BELOW, area_above=AREA_CLIPPING_ABOVE,
                                                  dwell_time_above=DWELL_TIME_CLIPPING_ABOVE,
                                                  visits_above=HOURLY_VISITS_CLIPPING_ABOVE,
                                                  inplace=True, min_pois_in_cat=100):

    attr_cols = ['safegraph_computed_area_in_square_feet', 'avg_median_dwell']
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    attr_cols.extend(hour_cols)
    assert all([col in d.columns for col in attr_cols])
    print('Clipping areas: %s (below=%d, above=%d), clipping dwell times: %s (above=%d), clipping visits: %s (above=%d)' %
          (clip_areas, area_below, area_above, clip_dwell_times, dwell_time_above, clip_visits, visits_above))
    
    assert 'sub_category' in d.columns
    category2idx = d.groupby('sub_category').indices
    category2idx = {cat:idx for cat,idx in category2idx.items() if len(idx) >= min_pois_in_cat}
    categories = sorted(category2idx.keys())
    print('Found %d categories with >= %d POIs' % (len(categories), min_pois_in_cat))
    new_data = np.array([d[col].copy() for col in attr_cols]).T  # n_pois x n_cols_to_clip
    thresholds = np.zeros((len(categories), len(attr_cols)+1))  # clipping thresholds for category x attribute
    medians = np.zeros((len(categories), len(attr_cols)))  # medians for category x attribute
    filled_dwell_time_nans = 0
    for i, cat in enumerate(categories):
        cat_idx = category2idx[cat]
        first_col_idx = 0  # index of first column for this attribute
        cat_areas = new_data[cat_idx, first_col_idx]
        min_area = np.nanpercentile(cat_areas, area_below)
        max_area = np.nanpercentile(cat_areas, area_above)
        median_area = np.nanmedian(cat_areas)
        thresholds[i][first_col_idx] = min_area
        thresholds[i][first_col_idx+1] = max_area
        medians[i][first_col_idx] = median_area
        nan_pois = cat_idx[np.isnan(cat_areas)]
        new_data[nan_pois, first_col_idx] = median_area
        if clip_areas:
            new_data[cat_idx, first_col_idx] = np.clip(cat_areas, min_area, max_area)
        first_col_idx += 1
        
        cat_dwell_times = new_data[cat_idx, first_col_idx]
        max_dwell_time = np.nanpercentile(cat_dwell_times, dwell_time_above)
        median_dwell_time = np.nanmedian(cat_dwell_times)
        thresholds[i][first_col_idx+1] = max_dwell_time
        medians[i][first_col_idx] = median_dwell_time
        nan_pois = cat_idx[np.isnan(cat_dwell_times)]
        filled_dwell_time_nans += len(nan_pois)
        new_data[nan_pois, first_col_idx] = median_dwell_time
        if clip_dwell_times:
            new_data[cat_idx, first_col_idx] = np.clip(cat_dwell_times, None, max_dwell_time)
        first_col_idx += 1
        
        col_idx = np.arange(first_col_idx, first_col_idx+len(hour_cols))
        assert col_idx[-1] == (len(attr_cols)-1)
        orig_visits = new_data[cat_idx][:, col_idx].copy()  # need to copy bc will modify
        orig_visit_sum = np.nansum(orig_visits)
        orig_visits[orig_visits == 0] = np.nan  # want percentile over positive visits
        # can't take percentile of col if it is all 0's or all nan's
        cols_to_process = col_idx[np.sum(~np.isnan(orig_visits), axis=0) > 0]
        max_visits_per_hour = np.nanpercentile(orig_visits[:, cols_to_process-first_col_idx], visits_above, axis=0)
        assert np.sum(np.isnan(max_visits_per_hour)) == 0
        thresholds[i][cols_to_process + 1] = max_visits_per_hour
        medians[i][cols_to_process] = np.nanmedian(orig_visits[:, cols_to_process-first_col_idx], axis=0)
        if clip_visits:
            orig_attributes = new_data[cat_idx]  # return to un-modified version
            orig_attributes[:, cols_to_process] = np.clip(orig_attributes[:, cols_to_process], None, max_visits_per_hour)
            new_data[cat_idx] = orig_attributes
        new_visit_sum = np.nansum(new_data[cat_idx][:, col_idx])
        print('%s -> found %d POIs, %d total visits before clipping, %d total visits after clipping' % 
              (cat, len(cat_idx), orig_visit_sum, new_visit_sum))
    print('Filled in avg_median_dwell for %d POIs that had nans before' % filled_dwell_time_nans)
    if inplace:
        new_d = d.copy()
        new_d[attr_cols] = new_data
        return new_d, categories, attr_cols, thresholds, medians
    new_d = pd.DataFrame(new_data, columns=attr_cols)
    new_d['sub_category'] = d['sub_category'].copy()
    return new_d, categories, attr_cols, thresholds

def apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list, poi_categories, poi_areas,
                                            new_all_hours, intervention_hour_idx,
                                            alpha, extra_weeks_to_simulate,
                                            top_category=None, sub_category=None,
                                            interpolate=True):
    # find POIs of interest
    if top_category is not None:
        top_category_poi_idx = (poi_categories['top_category'] == top_category).values
    else:
        top_category = 'any'
        top_category_poi_idx = np.ones(len(poi_categories)).astype(bool)
    if sub_category is not None:
        sub_category_poi_idx = (poi_categories['sub_category'] == sub_category).values
    else:
        sub_category = 'any'
        sub_category_poi_idx = np.ones(len(poi_categories)).astype(bool)
    intervened_poi_idx = top_category_poi_idx & sub_category_poi_idx  # poi indices to intervene on
    assert intervened_poi_idx.sum() > 0
    print("Intervening on POIs with top_category=%s, sub_category=%s (n=%i)" % (top_category, sub_category, intervened_poi_idx.sum()))

    # extend matrix list to extra weeks, loop final week for now
    num_pois, num_cbgs = poi_cbg_visits_list[0].shape
    new_matrix_list = [m.copy() for m in poi_cbg_visits_list]
    for i in range(extra_weeks_to_simulate * 168):
        matrix_idx = -168 + (i % 168)  # get corresponding matrix from final week
        new_matrix_list.append(poi_cbg_visits_list[matrix_idx].copy())
        assert new_matrix_list[-1].shape == (num_pois, num_cbgs), len(new_matrix_list)-1
    assert len(new_matrix_list) == len(new_all_hours)

    if top_category == 'any' and sub_category == 'any':  # apply intervention to all POIs
        full_activity_sum = 0
        simulated_activity_sum = 0
        for i in range(intervention_hour_idx, len(new_all_hours)):
            no_reopening = new_matrix_list[i]
            full_reopening = new_matrix_list[i % 168]
            full_activity_sum += full_reopening.sum()
            if alpha == 1:
                new_matrix_list[i] = full_reopening.copy()
                simulated_activity_sum = full_activity_sum
            else:
                if interpolate:
                    new_matrix_list[i] = full_reopening.multiply(alpha) + no_reopening.multiply(1-alpha)
                else:
                    new_matrix_list[i] = full_reopening.multiply(alpha)
                simulated_activity_sum += new_matrix_list[i].sum()
        diff = full_activity_sum - simulated_activity_sum
        overall_cost = (100. * diff / full_activity_sum)
        print('Overall Cost (%% of full activity): %2.3f%%' % overall_cost)
        return new_matrix_list, {'overall_cost':overall_cost, 'cost_within_intervened_pois':overall_cost}

    # full activity based on first week of visits
    range_end = max(intervention_hour_idx + 168, len(poi_cbg_visits_list))
    full_activity = [poi_cbg_visits_list[i % 168] for i in range(intervention_hour_idx, range_end)]  # get corresponding matrix in first week
    full_activity = hstack(full_activity, format='csr')
    orig_activity = hstack(new_matrix_list[intervention_hour_idx:range_end], format='csr')
    assert full_activity.shape == orig_activity.shape
    print('Computed hstacks of sparse matrices [shape=(%d, %d)]' % full_activity.shape)

    # take mixture of full activity and original activity for POIs of interest
    indicator_vec = np.zeros(num_pois)
    indicator_vec[intervened_poi_idx] = 1.0
    alpha_vec = alpha * indicator_vec
    scaled_full_activity = full_activity.transpose().multiply(alpha_vec).transpose()
    if interpolate:
        non_alpha_vec = 1.0 - alpha_vec   # intervened POIs will have alpha*full + (1-alpha)*closed
    else:
        non_alpha_vec = 1.0 - indicator_vec  # intervened POIs will have alpha*full
    scaled_orig_activity = orig_activity.transpose().multiply(non_alpha_vec).transpose()
    activity_mixture = scaled_full_activity + scaled_orig_activity
    print('Computed mixture of full and original activity')

    # compute costs
    full_overall_sum = full_activity.sum()
    mixture_overall_sum = activity_mixture.sum()
    overall_diff = full_overall_sum - mixture_overall_sum
    overall_cost = (100. * overall_diff / full_overall_sum)
    print('Overall Cost (%% of full activity): %2.3f%%' % overall_cost)
    full_intervened_sum = full_activity.transpose().multiply(indicator_vec).sum()
    mixture_intervened_sum = activity_mixture.transpose().multiply(indicator_vec).sum()
    intervened_diff = full_intervened_sum - mixture_intervened_sum
    cost_within_intervened_pois = (100. * intervened_diff / full_intervened_sum)
    print('Cost within intervened POIs: %2.3f%%' % cost_within_intervened_pois)

    print('Redistributing stacked matrix into hourly pieces...')
    ts = time.time()
    looping = False
    for i in range(intervention_hour_idx, len(new_all_hours)):
        matrix_idx = i - intervention_hour_idx
        if i >= len(poi_cbg_visits_list) and matrix_idx >= 168:
            # once we are operating past the length of real data, the "original" matrix
            # is just the matrix from the last week of the real data for the corresponding
            # day, and if matrix_idx > 168, then the mixture for that corresponding day
            # has been computed already
            new_matrix_list[i] = new_matrix_list[i - 168].copy()
            if looping is False:
                print('Entering looping phase at matrix %d!' % matrix_idx)
                looping = True
        else:
            matrix_start = matrix_idx * num_cbgs
            matrix_end = matrix_start + num_cbgs
            new_matrix_list[i] = activity_mixture[:, matrix_start:matrix_end]
        assert new_matrix_list[i].shape == (num_pois, num_cbgs), 'intervention idx = %d, overall idx = %d [found size = (%d, %d)]' % (matrix_idx, i, new_matrix_list[i].shape[0], new_matrix_list[i].shape[1])
        if matrix_idx % 24 == 0:
            te = time.time()
            print('Finished matrix %d: time so far per hourly matrix = %.2fs' % (matrix_idx, (te-ts)/(matrix_idx+1)))
    return new_matrix_list, {'overall_cost':overall_cost, 'cost_within_intervened_pois':cost_within_intervened_pois}

def get_ipf_filename(msa_name, min_datetime, max_datetime, clip_visits, correct_visits=True):
    fn = '%s_%s_to_%s_clip_visits_%s' % (msa_name,
                                min_datetime.strftime('%Y-%m-%d'),
                                max_datetime.strftime('%Y-%m-%d'),
                                clip_visits)
    if correct_visits:
        fn += '_correct_visits_True'
    filename = os.path.join(PATH_TO_IPF_OUTPUT, '%s.pkl' % fn)
    return filename

def sanity_check_error_metrics(fast_to_load_results):
    """
    Make sure train and test loss sum to total loss in the way we would expect.
    """
    n_train_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['train_loss_dict']['eval_start_time_cases'],
        fast_to_load_results['train_loss_dict']['eval_end_time_cases']))

    n_test_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['test_loss_dict']['eval_start_time_cases'],
        fast_to_load_results['test_loss_dict']['eval_end_time_cases']))

    n_total_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['loss_dict']['eval_start_time_cases'],
        fast_to_load_results['loss_dict']['eval_end_time_cases']))

    assert n_train_days + n_test_days == n_total_days
    assert fast_to_load_results['loss_dict']['eval_end_time_cases'] == fast_to_load_results['test_loss_dict']['eval_end_time_cases']
    assert fast_to_load_results['loss_dict']['eval_start_time_cases'] == fast_to_load_results['train_loss_dict']['eval_start_time_cases']
    for key in ['daily_cases_MSE', 'cumulative_cases_MSE']:
        if 'RMSE' in key:
            train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] ** 2 +
                 n_test_days * fast_to_load_results['test_loss_dict'][key] ** 2)

            overall_loss = n_total_days * fast_to_load_results['loss_dict'][key] ** 2
        else:
            train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] +
                 n_test_days * fast_to_load_results['test_loss_dict'][key])

            overall_loss = n_total_days * fast_to_load_results['loss_dict'][key]

        assert np.allclose(train_plus_test_loss, overall_loss, rtol=1e-6)
    print("Sanity check error metrics passed")

def deduplicate_grid_search_jobs(df, params_to_deduplicate_by, really_delete_stuff=False):
    """
    Sometimes we accidentally fit multiple grid search models with the same params - this deduplicates grid search results.
    It also deletes the duplicate models so they don't have problems anywhere.
    """
    old_len = len(df)
    # make sure jobs with the same params have the same results
    # both to confirm lack of stochasticity and to make sure other results didn't change.
    assert(df.groupby(params_to_deduplicate_by)['loss_dict_cumulative_cases_RMSE'].nunique().max() == 1)
    duplicate_idxs = df.duplicated(subset=params_to_deduplicate_by)
    if duplicate_idxs.sum() == 0:
        print("No duplicate files; returning.")
        return
    duplicated_timestrings = sorted(list(df.loc[duplicate_idxs, 'timestring'].values))
    print("Using %s, %i/%i rows are duplicates and will be deleted" % (','.join(params_to_deduplicate_by), len(duplicated_timestrings), old_len))
    for timestring in duplicated_timestrings:
        print("Deleting timestring %s" % timestring)
        paths_to_delete = [os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring),
                           os.path.join(helper.FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring),
                           os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring),
                           os.path.join(helper.FITTED_MODEL_DIR, 'full_models', 'fitted_model_%s.pkl' % timestring)]
        for path in paths_to_delete:
            assert os.path.exists(path)
            cmd = 'rm -f %s' % path
            if really_delete_stuff:
                os.system(cmd)
                assert not os.path.exists(path)
            else:
                print(cmd)
    df = df.loc[~duplicate_idxs]
    df.index = range(len(df))
    print("After removing duplicates, new length of df is %i" % len(df))
    return df

def fit_and_save_one_model(timestring,
                           model_kwargs,
                           data_kwargs,
                           d=None,
                           experiment_to_run=None,
                           train_test_partition=None):
    '''
    Fits one model, saves its results and evaluations of the results.
    timestring: the string to use in filenames to identify the model and its config;
                if None, then the model is not saved
    model_kwargs: arguments to use for fit_disease_model_on_real_data
    data_kwargs: arguments for the data; must have the field 'MSA_name'
    d: the dataframe for the MSA pois; if it is None, then the dataframe is loaded
       within the function
    train_test_partition: the first hour of test; if included, then losses are saved
                          separately for train and test dates
    '''
    t0 = time.time()
    return_without_saving = False
    if timestring is None:
        print("Fitting single model. Timestring is none so not saving model and just returning fitted model.")
        return_without_saving = True
    else:
        print("Fitting single model. Results will be saved using timestring %s" % timestring)
    if d is None:  # load data
        d = helper.load_dataframe_for_individual_msa(**data_kwargs)
    assert('MSA_name' in data_kwargs)
    nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(data_kwargs['MSA_name'])
    if 'counties_to_track' not in model_kwargs:
        model_kwargs['counties_to_track'] = msa_counties
    cbg_groups_to_track = {}
    cbg_groups_to_track['nyt'] = nyt_cbgs
    if ('filter_for_cbgs_in_msa' in model_kwargs) and model_kwargs['filter_for_cbgs_in_msa']:
        print("Filtering for %i CBGs within MSA %s" % (len(msa_cbgs), data_kwargs['MSA_name']))
        cbgs_to_filter_for = set(msa_cbgs) # filter for CBGs within MSA
    else:
        cbgs_to_filter_for = None

    if experiment_to_run == 'just_save_ipf_output':
        # If we're saving IPF output, don't try to reload file.
        preload_poi_visits_list_filename = None
    elif 'poi_cbg_visits_list' in model_kwargs:
        print('Passing in poi_cbg_visits_list, will not load from file')
        preload_poi_visits_list_filename = None
    else:
        # Otherwise, default to attempting to load file.
        preload_poi_visits_list_filename = get_ipf_filename(msa_name=data_kwargs['MSA_name'],
            min_datetime=model_kwargs['min_datetime'],
            max_datetime=model_kwargs['max_datetime'],
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'])
        if not os.path.exists(preload_poi_visits_list_filename):
            print("Warning: path %s does not exist; regenerating POI visits" % preload_poi_visits_list_filename)
            preload_poi_visits_list_filename = None
        else:
            print("Reloading POI visits from %s" % preload_poi_visits_list_filename)
    model_kwargs['preload_poi_visits_list_filename'] = preload_poi_visits_list_filename

    # fit model
    fitted_model = fit_disease_model_on_real_data(
        d,
        cbg_groups_to_track=cbg_groups_to_track,
        cbgs_to_filter_for=cbgs_to_filter_for,
        **model_kwargs)

    if experiment_to_run == 'just_save_ipf_output':
        pickle_start_time = time.time()
        ipf_filename = get_ipf_filename(msa_name=data_kwargs['MSA_name'],
            min_datetime=model_kwargs['min_datetime'],
            max_datetime=model_kwargs['max_datetime'],
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'],
            correct_visits=model_kwargs['correct_poi_visits'])
        print('saving IPF output in', ipf_filename)
        ipf_file = open(ipf_filename, 'wb')
        pickle.dump(fitted_model.poi_cbg_visit_history, ipf_file)
        ipf_file.close()
        print('time to save pickle = %.2fs' % (time.time() - pickle_start_time))
        print('size of pickle: %.2f MB' % (os.path.getsize(ipf_filename) / (1024**2)))
        return

    if return_without_saving:
        return fitted_model

    # Save model
    mdl_path = os.path.join(helper.FITTED_MODEL_DIR, 'full_models', 'fitted_model_%s.pkl' % timestring)
    print("Saving model at %s..." % mdl_path)
    file = open(mdl_path, 'wb')
    fitted_model.save(file)
    file.close()

    # evaluate results broken down by race and SES.
    plot_path = os.path.join(helper.FITTED_MODEL_DIR, 'ses_race_plots', 'ses_race_plot_%s.pdf' % timestring)
    ses_race_results = make_slir_race_ses_plot(fitted_model, path_to_save=plot_path)

    # Save some smaller model results for quick(er) loading. For really fast stuff, like losses (numerical results only) we store separately.
    print("Saving model results...")
    file = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'wb')
    model_results_to_save_separately = {}
    for attr_to_save_separately in ['history', 'CBGS_TO_IDXS']:
        model_results_to_save_separately[attr_to_save_separately] = getattr(fitted_model, attr_to_save_separately)
    model_results_to_save_separately['ses_race_results'] = ses_race_results
    pickle.dump(model_results_to_save_separately, file)
    file.close()

    # evaluate model fit to cases and save loss separately as well.
    # Everything saved in this data structure should be a summary result - small and fast to load, numbers only!
    loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           model_results=model_results_to_save_separately)
    fast_to_load_results = {'loss_dict':loss_dict}
    if train_test_partition is not None:
        train_max = train_test_partition + datetime.timedelta(hours=-1)
        train_loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_end_time = train_max,
                                           model_results=model_results_to_save_separately)
        fast_to_load_results['train_loss_dict'] = train_loss_dict
        test_loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_start_time = train_test_partition,
                                           model_results=model_results_to_save_separately)
        fast_to_load_results['test_loss_dict'] = test_loss_dict
        fast_to_load_results['train_test_date_cutoff'] = train_test_partition
        sanity_check_error_metrics(fast_to_load_results)

    fast_to_load_results['clipping_monitor'] = fitted_model.clipping_monitor
    fast_to_load_results['final infected fraction'] = (fitted_model.cbg_infected + fitted_model.cbg_removed + fitted_model.cbg_latent).sum(axis=1)/fitted_model.CBG_SIZES.sum()
    fast_to_load_results['ses_race_summary_results'] = {}
    demographic_group_keys = ['p_black', 'p_white', 'median_household_income']
    for k1 in demographic_group_keys:
        for k2 in ['above_median', 'below_median', 'top_decile', 'bottom_decile', 'above_median_in_own_county', 'below_median_in_own_county']:
            full_key = 'L+I+R, %s_%s' % (k1, k2)
            fast_to_load_results['ses_race_summary_results']['final fraction in state ' + full_key] = ses_race_results[full_key][-1]
    fast_to_load_results['estimated_R0'] = fitted_model.estimated_R0
    fast_to_load_results['intervention_cost'] = fitted_model.INTERVENTION_COST

    for k1 in demographic_group_keys:
        for (top_group, bot_group) in [
            ('above_median', 'below_median'),
            ('top_decile', 'bottom_decile'),
            ('above_median_in_own_county', 'below_median_in_own_county')]:
            top_group_key = f'{k1}_{top_group}'
            bot_group_key = f'{k1}_{bot_group}'
            top_group_LIR_ratio = ((fitted_model.history[top_group_key]['latent'][:, -1] +
                             fitted_model.history[top_group_key]['infected'][:, -1] +
                             fitted_model.history[top_group_key]['removed'][:, -1]) /
                             fitted_model.history[top_group_key]['total_pop'])
            bot_group_LIR_ratio = ((fitted_model.history[bot_group_key]['latent'][:, -1] +
                             fitted_model.history[bot_group_key]['infected'][:, -1] +
                             fitted_model.history[bot_group_key]['removed'][:, -1]) /
                             fitted_model.history[bot_group_key]['total_pop'])
            fast_to_load_results['ses_race_summary_results'][f'{k1}_{bot_group}_over_{top_group}_L+I+R_ratio_fixed'] = bot_group_LIR_ratio / top_group_LIR_ratio

    file = open(os.path.join(helper.FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()

    # Save kwargs.
    data_and_model_kwargs = {'model_kwargs':model_kwargs, 'data_kwargs':data_kwargs, 'experiment_to_run':experiment_to_run}
    file = open(os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'wb')
    pickle.dump(data_and_model_kwargs, file)
    file.close()
    print("Successfully fitted and saved model and data_and_model_kwargs; total time taken %2.3f seconds" % (time.time() - t0))
    return fitted_model

def load_model_and_data_from_timestring(timestring, verbose=False, load_original_data=False,
                                        load_full_model=False, load_fast_results_only=True,
                                        load_filtered_data_model_was_fitted_on=False):
    if verbose:
        print("Loading model from timestring %s" % timestring)

    f = open(os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')
    data_and_model_kwargs = pickle.load(f)
    f.close()
    model = None
    model_results = None
    f = open(os.path.join(helper.FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'rb')
    fast_to_load_results = pickle.load(f)
    f.close()

    if not load_fast_results_only:
        f = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'rb')
        model_results = pickle.load(f)
        f.close()

        if load_full_model:
            f = open(os.path.join(helper.FITTED_MODEL_DIR, 'full_models', 'fitted_model_%s.pkl' % timestring), 'rb')
            model = pickle.load(f)
            f.close()

    if load_original_data:
        if verbose:
            print("Loading original data as well...warning, this may take a while")
        d = helper.load_dataframe_for_individual_msa(**data_and_model_kwargs['data_kwargs'])
    else:
        d = None

    if load_filtered_data_model_was_fitted_on:
        # if true, return the data after all the filtering, along with the model prior to fitting.

        data_kwargs = data_and_model_kwargs['data_kwargs'].copy()
        model_kwargs = data_and_model_kwargs['model_kwargs'].copy()
        model_kwargs['return_model_and_data_without_fitting'] = True
        unfitted_model = fit_and_save_one_model(timestring=None,
                                     model_kwargs=model_kwargs,
                                     data_kwargs=data_kwargs,
                                     train_test_partition=None)
        filtered_data = unfitted_model.d
        return model, data_and_model_kwargs, d, model_results, fast_to_load_results, filtered_data, unfitted_model

    else:
        return model, data_and_model_kwargs, d, model_results, fast_to_load_results



def filter_timestrings_for_properties(required_properties=None,
                                      required_model_kwargs=None,
                                      required_data_kwargs=None,
                                      min_timestring=None,
                                      max_timestring=None,
                                      return_msa_names=False):
    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs
    """

    if required_properties is None:
        required_properties = {}
    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_data_kwargs is None:
        required_data_kwargs = {}
    if max_timestring is None:
        max_timestring = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
    print("Loading models with timestrings between %s and %s" % (str(min_timestring), max_timestring))
    config_dir = os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs')
    matched_timestrings = []
    msa_names = []

    for fn in os.listdir(config_dir):
        if fn.startswith('config_'):
            timestring = fn.lstrip('config_').rstrip('.pkl')
            if (timestring < max_timestring) and (min_timestring is None or timestring >= min_timestring):
                f = open(os.path.join(config_dir, fn), 'rb')
                data_and_model_kwargs = pickle.load(f)
                f.close()
                if test_if_kwargs_match(required_properties,
                                        required_data_kwargs,
                                        required_model_kwargs,
                                        data_and_model_kwargs):
                    matched_timestrings.append(timestring)
                    msa_names.append(data_and_model_kwargs['data_kwargs']['MSA_name'])
    if not return_msa_names:
        return matched_timestrings
    else:
        return matched_timestrings, msa_names

    return matched_timestrings

def test_if_kwargs_match(req_properties, req_data_kwargs,
                         req_model_kwargs, test_data_and_model_kwargs):
    # check whether direct properties in test_data_and_model_kwargs match
    prop_match = all([req_properties[key] == test_data_and_model_kwargs[key] for key in req_properties if key not in ['data_kwargs', 'model_kwargs']])
    if not prop_match:
        return False

    # check whether data kwargs match
    test_data_kwargs = test_data_and_model_kwargs['data_kwargs']
    data_match = all([req_data_kwargs[key] == test_data_kwargs[key] for key in req_data_kwargs])
    if not data_match:
        return False

    # check if non-dictionary model kwargs match
    kwargs_keys = set([key for key in req_model_kwargs if 'kwargs' in key])
    test_model_kwargs = test_data_and_model_kwargs['model_kwargs']
    model_match = all([req_model_kwargs[key] == test_model_kwargs[key] for key in req_model_kwargs if key not in kwargs_keys])
    if not model_match:
        return False

    # check if elements within dictionary model kwargs match
    for kw_key in kwargs_keys:
        req_kwargs = req_model_kwargs[kw_key]
        test_kwargs = test_model_kwargs[kw_key]
        kw_match = all([req_kwargs[k] == test_kwargs[k] for k in req_kwargs])
        if not kw_match:
            return False
    return True

def generate_data_and_model_configs(msas_to_fit=10,
                                    config_idx_to_start_at=None,
                                    skip_previously_fitted_kwargs=False,
                                    min_timestring=None,
                                    min_timestring_to_load_best_fit_models_from_grid_search=None,
                                    experiment_to_run='normal_grid_search',
                                    max_models_to_take_per_msa=MAX_MODELS_TO_TAKE_PER_MSA,
                                    acceptable_loss_tolerance=ACCEPTABLE_LOSS_TOLERANCE):
    """
    MSAs to fit: how many MSAs we will focus on.
    config_idx_to_start_at: how many configs we should skip.
    """
    # this controls what parameters we search over.
    config_generation_start_time = time.time()

    if skip_previously_fitted_kwargs:
        assert min_timestring is not None
        previously_fitted_timestrings = filter_timestrings_for_properties(min_timestring=min_timestring)
        previously_fitted_data_and_model_kwargs = [pickle.load(open(os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')) for timestring in previously_fitted_timestrings]
        print("Filtering out %i previously generated configs" % len(previously_fitted_data_and_model_kwargs))
    else:
        previously_fitted_data_and_model_kwargs = []

    # Helper dataframe to check current status of data
    d = helper.load_chunk(1, load_backup=False)

    # Data kwargs
    data_kwargs = []

    # Load on largest N MSAs.
    biggest_msas = d['poi_lat_lon_CBSA Title'].value_counts().head(n=msas_to_fit)

    print("Largest %i MSAs are" % len(biggest_msas))
    print(biggest_msas)

    for msa_name in biggest_msas.index:
        name_without_spaces = re.sub('[^0-9a-zA-Z]+', '_', msa_name)
        data_kwargs.append({'MSA_name':name_without_spaces, 'nrows':None})

    # Now generate model kwargs.
    # min_datetime and max_datetime are fixed
    min_datetime = datetime.datetime(2020, 3, 1, 0)
    date_cols = [helper.load_date_col_as_date(a) for a in d.columns]
    date_cols = [a for a in date_cols if a is not None]
    max_date = max(date_cols)
    max_datetime = datetime.datetime(max_date.year, max_date.month, max_date.day, 23)  # latest hour
    print('Min datetime: %s. Max datetime: %s.' % (min_datetime, max_datetime))
    use_density_based_home_rates = False

    # Generate model kwargs. How exactly we do this depends on which experiments we're running.
    num_seeds = 20
    configs_with_changing_params = []
    if experiment_to_run == 'just_save_ipf_output':
        # this is simpler - all we want to do is save the IPF output, once for each MSA.
#         model_kwargs = []
#         for clip_visits in [True, False]:
#             model_kwargs.append({'min_datetime':min_datetime,
#                         'max_datetime':max_datetime,
#                         'exogenous_model_kwargs': {
#                             'home_beta':1e-2,
#                             'poi_psi':1000,
#                             'p_sick_at_t0':1e-4,
#                             'just_compute_r0':False,
#                             },
#                         'model_init_kwargs':{
#                             'starting_seed':0,
#                             'ipf_final_match':'poi',
#                             'clip_poisson_approximation':True,
#                             'approx_method':'poisson'
#                             },
#                           'poi_attributes_to_clip':{'clip_areas':True,
#                                           'clip_dwell_times':True,
#                                           'clip_visits':clip_visits},
#                           'correct_poi_visits':True,
#                          'simulation_kwargs':{'verbosity':24},
#                          'include_cbg_prop_out':True})
        model_kwargs = [{'min_datetime':min_datetime,
                         'max_datetime':max_datetime,
                         'exogenous_model_kwargs': {
                            'home_beta':1e-2,
                            'poi_psi':1000,
                            'p_sick_at_t0':1e-4,
                            'just_compute_r0':False,
                            },
                         'model_init_kwargs':{
                            'starting_seed':0,
                            'ipf_final_match':'poi',
                            'clip_poisson_approximation':True,
                            'approx_method':'poisson'
                            },
                          'poi_attributes_to_clip':{'clip_areas':True,
                                          'clip_dwell_times':True,
                                          'clip_visits':True},
                          'correct_poi_visits':True,
                          'simulation_kwargs':{'verbosity':24},
                          'include_cbg_prop_out':True}]
    elif experiment_to_run == 'normal_grid_search':
        p_sicks = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]#np.array([10 ** a for a in np.arange(-6, -4, .2)])#[1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
        home_betas = [0.001, 0.002, 0.005, 0.008, 0.01, 0.012]#[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        #home_psis = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
        poi_psis = [a for a in np.arange(200, 4101, 300)]#[a for a in np.arange(1000, 10001, 500)]


        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    configs_with_changing_params.append({'home_beta':home_beta, 'home_psi':None, 'poi_psi':poi_psi, 'p_sick_at_t0':p_sick, 'ipf_final_match':'poi', 'clip_poisson_approximation':True})

        # ablation analyses.
        for home_beta in sorted(list(set(home_betas + list(np.arange(0.0001, 0.002, 0.0001)) + [0.0015, 0.003, 0.004]))):
            for p_sick in p_sicks:
                configs_with_changing_params.append({'home_beta':home_beta, 'home_psi':None, 'poi_psi':0, 'p_sick_at_t0':p_sick, 'ipf_final_match':'poi', 'clip_poisson_approximation':True})


    elif experiment_to_run == 'calibrate_r0':
        poi_psis = [20000, 15000, 10000,7500,6000, 5000,4500,4000,3500,3000,2500,2000,1500,1000,500,250,100]#[1e5, 5e4, 1e4, 5e3, 1e3, 0]
        home_betas = [5e-2, 0.02, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 0]
        for home_beta in home_betas:
            configs_with_changing_params.append({'home_beta':home_beta, 'poi_psi':2500, 'p_sick_at_t0':1e-4, 'ipf_final_match':'poi', 'clip_poisson_approximation':True, 'home_psi':None})
        for poi_psi in poi_psis:
            configs_with_changing_params.append({'home_beta':0.02, 'poi_psi':poi_psi, 'p_sick_at_t0':1e-4, 'ipf_final_match':'poi', 'clip_poisson_approximation':True, 'home_psi':None})

    # experiments that require the best fit models
    best_models_experiments = {'test_interventions', 'test_retrospective_counterfactuals', 'test_max_capacity_clipping',
                               'test_uniform_interpolated_reopening', 'test_uniform_proportion_of_full_reopening',
                               'rerun_best_models_and_save_cases_per_poi'}
    if experiment_to_run in best_models_experiments:
        # here model and data kwargs are entwined, so we can't just take the outer product of model_kwargs and data_kwargs.
        # this is because we load the best fitting model for each MSA.
        list_of_data_and_model_kwargs = []
        poi_categories_to_examine = 30
        key_to_sort_by = 'loss_dict_daily_cases_RMSE'

        # examine most common POI categories.
        most_visited_poi_subcategories = get_list_of_poi_subcategories_with_most_visits(n_poi_categories=poi_categories_to_examine)

        # Get list of all fitted models.
        model_timestrings, model_msas = filter_timestrings_for_properties(
            min_timestring=min_timestring_to_load_best_fit_models_from_grid_search,
            return_msa_names=True)
        print("Found %i models after %s" % (len(model_timestrings), min_timestring_to_load_best_fit_models_from_grid_search))
        timestring_msa_df = pd.DataFrame({'model_timestring':model_timestrings, 'model_msa':model_msas})
        for row in data_kwargs:
            msa_t0 = time.time()
            msa_name = row['MSA_name']
            timestrings_for_msa = timestrings=list(
                timestring_msa_df.loc[timestring_msa_df['model_msa'] == msa_name, 'model_timestring'].values)
            print("Evaluating %i timestrings for %s" % (len(timestrings_for_msa), msa_name))
            best_msa_models = evaluate_all_fitted_models_for_msa(msa_name, timestrings=timestrings_for_msa)
            best_msa_models = best_msa_models.loc[(best_msa_models['experiment_to_run'] == 'normal_grid_search') & (best_msa_models['poi_psi'] > 0)].sort_values(by=key_to_sort_by)

            best_loss = float(best_msa_models.iloc[0][key_to_sort_by])
            print("After filtering for normal_grid_search models, %i models for MSA" % (len(best_msa_models)))
            best_msa_models = best_msa_models.loc[best_msa_models[key_to_sort_by] <= acceptable_loss_tolerance * best_loss]

            best_msa_models = best_msa_models.iloc[:max_models_to_take_per_msa]
            print("After filtering for models with %s within factor %2.3f of best loss, and taking max %i models, %i models" %
                (key_to_sort_by, acceptable_loss_tolerance, max_models_to_take_per_msa, len(best_msa_models)))

            for i in range(len(best_msa_models)):
                loss_ratio = best_msa_models.iloc[i][key_to_sort_by]/best_loss
                assert loss_ratio >= 1 and loss_ratio <= acceptable_loss_tolerance
                model_quality_dict = {'model_fit_rank_for_msa':i,
                                          'ratio_of_%s_to_that_of_best_fitting_model' % key_to_sort_by:loss_ratio,
                                          'model_timestring':best_msa_models.iloc[i]['timestring']}
                _, kwargs_i, _, _, _ = load_model_and_data_from_timestring(best_msa_models.iloc[i]['timestring'], load_fast_results_only=True)
                kwargs_i['experiment_to_run'] = experiment_to_run
                del kwargs_i['model_kwargs']['counties_to_track']
                kwargs_i['model_kwargs']['include_cbg_prop_out'] = False
                if 'include_cbg_reweighting' in kwargs_i['model_kwargs']:
                    del kwargs_i['model_kwargs']['include_cbg_reweighting']

                if experiment_to_run == 'test_retrospective_counterfactuals':
                    # LOOKING AT THE PAST.
                    # what if we had only done x% of social distancing?
                    # degree represents what percentage of social distancing to keep - we don't need to test 1
                    # because that is what actually happened
                    for degree in [0, 0.25, 0.5, 0.75]:
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'distancing_degree':degree}
                        counterfactual_retrospective_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                    # what if we shifted the timeseries by x days?
                    for shift in [-14, -7, -3, 3, 7, 14]:  # how many days to shift
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'shift_in_days':shift}
                        counterfactual_retrospective_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)
                elif experiment_to_run == 'test_interventions':
                    # FUTURE EXPERIMENTS: reopen each category of POI.
                    for cat_idx in range(len(most_visited_poi_subcategories)):
                        for alpha in [0, 1]:
                            kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                            counterfactual_poi_opening_experiment_kwargs = {'alpha':alpha,
                                                   'extra_weeks_to_simulate':4,
                                                   'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                                   'top_category':None,
                                                   'sub_category':most_visited_poi_subcategories[cat_idx]}
                            counterfactual_poi_opening_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()

                            kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                            list_of_data_and_model_kwargs.append(kwarg_copy)
                elif experiment_to_run == 'test_max_capacity_clipping':
                    # FUTURE EXPERIMENTS: reopen fully but clip at alpha-proportion of max capacity
                    for alpha in np.arange(.1, 1, .1):
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                                               'extra_weeks_to_simulate':4,
                                               'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                               'alpha':1,
                                               'max_capacity_alpha':alpha}
                        counterfactual_poi_opening_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)
                elif experiment_to_run == 'test_uniform_interpolated_reopening':
                    # FUTURE EXPERIMENTS: test uniform reopening on all pois, interpolating between pre- and post-
                    # lockdown activity
                    for alpha in np.arange(0, 1.1, .1):
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                                               'extra_weeks_to_simulate':4,
                                               'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                               'alpha':alpha}
                        counterfactual_poi_opening_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)
                elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
                    # FUTURE EXPERIMENTS: test uniform reopening on all pois, simple proportion of pre-lockdown activity
                    for alpha in np.arange(.3, 1.1, .1):
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                                               'extra_weeks_to_simulate':4,
                                               'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                               'full_activity_alpha':alpha}
                        counterfactual_poi_opening_experiment_kwargs['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)
                else:
                    kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                    simulation_kwargs = {'groups_to_track_num_cases_per_poi':['all', 'median_household_income_bottom_decile',
                                                                         'median_household_income_top_decile']}
                    kwarg_copy['model_kwargs']['simulation_kwargs'] = simulation_kwargs
                    list_of_data_and_model_kwargs.append(kwarg_copy)
            print("In total, it took %2.3f seconds to generate configs for MSA" % (time.time() - msa_t0))

        # sanity check to make sure nothing strange - number of parameters we expect.
        make_sure_nothing_weird_happened = []

        for row in list_of_data_and_model_kwargs:
            make_sure_nothing_weird_happened.append(
                {'home_beta':row['model_kwargs']['exogenous_model_kwargs']['home_beta'],
                 'poi_psi':row['model_kwargs']['exogenous_model_kwargs']['poi_psi'],
                 'p_sick_at_t0':row['model_kwargs']['exogenous_model_kwargs']['p_sick_at_t0'],
                 'MSA_name':row['data_kwargs']['MSA_name']})
        make_sure_nothing_weird_happened = pd.DataFrame(make_sure_nothing_weird_happened)

        n_experiments_per_param_setting = make_sure_nothing_weird_happened.groupby(['home_beta',
                                                  'poi_psi',
                                                  'p_sick_at_t0',
                                                 'MSA_name']).size()
        if experiment_to_run == 'test_interventions':
            assert (n_experiments_per_param_setting.values == poi_categories_to_examine * 2).all()
        elif experiment_to_run == 'test_max_capacity_clipping':
            assert (n_experiments_per_param_setting.values == 9).all()
        elif experiment_to_run == 'test_uniform_interpolated_reopening':
            assert (n_experiments_per_param_setting.values == 11).all()
        elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
            assert (n_experiments_per_param_setting.values == 8).all()
        elif experiment_to_run == 'rerun_best_models_and_save_cases_per_poi':
            assert (n_experiments_per_param_setting.values == 1).all()
        else:
            assert (n_experiments_per_param_setting.values == 10).all()

    else:
        if experiment_to_run != 'just_save_ipf_output':
            model_kwargs = []
            for config in configs_with_changing_params:
                model_kwargs.append({'min_datetime':min_datetime,
                                     'max_datetime':max_datetime,
                                     'cbg_count_cutoff':10,
                                     'exogenous_model_kwargs': {
                                        'home_beta':config['home_beta'],
                                        'poi_psi':config['poi_psi'],
                                        'home_psi':config['home_psi'],
                                        'p_sick_at_t0':config['p_sick_at_t0'],
                                        'just_compute_r0':experiment_to_run=='calibrate_r0',
                                        },
                                     'model_init_kwargs':{
                                            'starting_seed':0,
                                            'num_seeds':num_seeds,
                                            'ipf_final_match':config['ipf_final_match'],
                                            'clip_poisson_approximation':config['clip_poisson_approximation'],
                                            'approx_method':'poisson'},
                                     'poi_attributes_to_clip':{'clip_areas':True,
                                          'clip_dwell_times':True,
                                          'clip_visits':True},
                                        'simulation_kwargs':{'verbosity':24},
                                     'verbose':True,
                                     'use_density_based_home_rates':use_density_based_home_rates,
                                     'include_cbg_prop_out':False, # not needed anymore because we're using preloading IPF output
                                     'filter_for_cbgs_in_msa':False})

        list_of_data_and_model_kwargs = [{'data_kwargs':copy.deepcopy(a), 'model_kwargs':copy.deepcopy(b), 'experiment_to_run':experiment_to_run} for b in model_kwargs for a in data_kwargs]

    # remove previously fitted kwargs"
    if len(previously_fitted_data_and_model_kwargs) > 0:
        print("Prior to filtering out previously fitted kwargs, %i kwargs" % len(list_of_data_and_model_kwargs))
        for i in range(len(previously_fitted_data_and_model_kwargs)):
            # remove stuff that is added when saved so configs are comparable.
            if 'counties_to_track' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
                del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['counties_to_track']
            if 'preload_poi_visits_list_filename' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
                del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['preload_poi_visits_list_filename']
        list_of_data_and_model_kwargs = [a for a in list_of_data_and_model_kwargs if a not in previously_fitted_data_and_model_kwargs]
        #list_of_data_and_model_kwargs = list_of_data_and_model_kwargs[len(previously_fitted_data_and_model_kwargs):]
        print("After removing previously fitted kwargs, %i kwargs" % (len(list_of_data_and_model_kwargs)))

    print("Total data/model configs to fit: %i; randomly shuffling order" % len(list_of_data_and_model_kwargs))
    random.Random(0).shuffle(list_of_data_and_model_kwargs)
    if config_idx_to_start_at is not None:
        print("Skipping first %i configs" % config_idx_to_start_at)
        list_of_data_and_model_kwargs = list_of_data_and_model_kwargs[config_idx_to_start_at:]
    print("Total time to generate configs: %2.3f seconds" % (time.time() - config_generation_start_time))
    return list_of_data_and_model_kwargs

def check_memory_usage():
    virtual_memory = psutil.virtual_memory()
    total_memory = getattr(virtual_memory, 'total')
    available_memory = getattr(virtual_memory, 'available')
    free_memory = getattr(virtual_memory, 'free')
    available_memory_percentage = 100. * available_memory / total_memory
    # Free memory is the amount of memory which is currently not used for anything. This number should be small, because memory which is not used is simply wasted.
    # Available memory is the amount of memory which is available for allocation to a new process or to existing processes.
    # Adrijan thinks we care about available.
    print('Total memory: %s; free memory: %s; available memory %s; available memory %2.3f%%' % (
        bytes2human(total_memory),
        bytes2human(free_memory),
        bytes2human(available_memory),
        available_memory_percentage))
    return available_memory_percentage

def run_many_models_in_parallel(configs_to_fit):
    CPU_USAGE_THRESHOLD = 70#70
    MEM_USAGE_THRESHOLD = 70#70
    MAX_PROCESSES_FOR_USER = int(multiprocessing.cpu_count() / 1.5)
    print("Maximum number of processes to run: %i" % MAX_PROCESSES_FOR_USER)
    SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD = 10
    for config_idx in range(len(configs_to_fit)):
        t0 = time.time()

        # Check how many processes user is running.
        n_processes_running = int(subprocess.check_output('ps -fA | grep model_experiments.py | wc -l', shell=True))
        print("Current processes running for user: %i" % n_processes_running)
        while n_processes_running > MAX_PROCESSES_FOR_USER:
            print("Current processes are %i, above threshold of %i; waiting." % (n_processes_running, MAX_PROCESSES_FOR_USER))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            n_processes_running = int(subprocess.check_output('ps -fA | grep model_experiments.py | wc -l', shell=True))

        # don't swamp cluster. Check CPU usage.
        cpu_usage = psutil.cpu_percent()
        print("Current CPU usage: %2.3f%%" % cpu_usage)
        while cpu_usage > CPU_USAGE_THRESHOLD:
            print("Current CPU usage is %2.3f, above threshold of %2.3f; waiting." % (cpu_usage, CPU_USAGE_THRESHOLD))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            cpu_usage = psutil.cpu_percent()

        # Also check memory.
        available_memory_percentage = check_memory_usage()
        while available_memory_percentage < 100 - MEM_USAGE_THRESHOLD:
            print("Current memory usage is above threshold of %2.3f; waiting." % (MEM_USAGE_THRESHOLD))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            available_memory_percentage = check_memory_usage()



        # If we pass these checks, start a job.
        timestring = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
        experiment_to_run = configs_to_fit[config_idx]['experiment_to_run']
        print("Starting job %i/%i" % (config_idx + 1, len(configs_to_fit)))
        outfile_path = os.path.join(helper.FITTED_MODEL_DIR, 'model_fitting_logfiles/%s.out' % timestring)
        cmd = 'nohup python -u model_experiments.py fit_and_save_one_model %s --timestring %s --config_idx %i > %s 2>&1 &' % (experiment_to_run, timestring, config_idx, outfile_path)
        print("Command: %s" % cmd)
        os.system(cmd)
        time.sleep(3)
        print("Time between job submissions: %2.3f" % (time.time() - t0))

###################################################
# Code for plotting and visualizations
###################################################

def invert_dwell_time_correction_factors(correction_factors):
    assert ((correction_factors <= 1) & (correction_factors >= 0)).all()
    original_time_in_minutes = 60 * np.sqrt(correction_factors)  / (1 - np.sqrt(correction_factors))
    return original_time_in_minutes

def unpack_random_seeds(df, cols_of_interest, cols_to_keep):
    """
    Small helper method: given a df, a column you should have random seeds for (col_of_interest) and a list of other columns you want to keep, turn it into a long df with one row for each random seed.
    """
    long_df = []
    for i in range(len(df)):
        df_for_row = {}
        for k in cols_to_keep:
            df_for_row[k] = df.iloc[i][k]
        for col_of_interest in cols_of_interest:
            vals_across_random_seeds = list(df.iloc[i][col_of_interest])
            df_for_row[col_of_interest] = vals_across_random_seeds
        df_for_row['random_seed'] = range(len(vals_across_random_seeds))
        df_for_row = pd.DataFrame(df_for_row)
        long_df.append(df_for_row)
    return pd.concat(long_df)


def make_boxplot_of_poi_reopening_effects(intervention_df, msa_names, poi_characteristics, titlestring, cats_to_plot, filename):
    subcategory_counts = {}
    poi_characteristics = copy.deepcopy(poi_characteristics)
    for k in poi_characteristics:
        poi_characteristics[k]['pretty_name'] = poi_characteristics[k]['sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)

    for k in poi_characteristics:
        subcategory_counts[k] = Counter(poi_characteristics[k]['pretty_name'])
    msa_names = [a for a in msa_names if a in poi_characteristics]
    print("Making plots using", msa_names)
    assert len(msa_names) > 0

    poi_characteristics_df = []
    for msa in msa_names:
        poi_characteristics_df.append(poi_characteristics[msa])
    poi_characteristics_df = pd.concat(poi_characteristics_df)
    poi_characteristics_df = poi_characteristics_df.loc[poi_characteristics_df['sub_category'].map(lambda x:x in cats_to_plot)]
    poi_characteristics_df['original_dwell_times'] = invert_dwell_time_correction_factors(poi_characteristics_df['dwell_time_correction_factors'].values)
    poi_characteristics_df['density*dwell_time_factor'] = poi_characteristics_df['dwell_time_correction_factors'] * poi_characteristics_df['weighted_visits_over_area']
    poi_characteristics_df['visits^2*dwell_time_factor/area'] = poi_characteristics_df['dwell_time_correction_factors'] * poi_characteristics_df['squared_visits_over_area']

    intervention_df = intervention_df.copy()
    intervention_df = intervention_df.loc[intervention_df['MSA_name'].map(lambda x:x in msa_names)]
    total_modeled_pops = {}
    for msa in msa_names:
        ts = intervention_df.loc[intervention_df['MSA_name'] == msa, 'timestring'].iloc[0]
        model, _, _, _, _ = load_model_and_data_from_timestring(
                ts, 
                load_fast_results_only=False,
                load_full_model=True)
        total_modeled_pops[msa] = model.CBG_SIZES.sum()


    #intervention_df['mean_final_infected_fraction'] = intervention_df['final infected fraction'].map(
    #    lambda x:np.mean(x))
    intervention_df['pretty_cat_names'] = intervention_df['counterfactual_sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x in cats_to_plot)]
    full_reopen_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 1]
    full_close_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 0]
    merge_cols = ['model_fit_rank_for_msa', 'pretty_cat_names', 'MSA_name']

    full_close_df = unpack_random_seeds(full_close_df,
        cols_of_interest=['final infected fraction'],
        cols_to_keep=merge_cols)
    full_reopen_df = unpack_random_seeds(full_reopen_df,
        cols_of_interest=['final infected fraction'],
        cols_to_keep=merge_cols)


    merge_cols = merge_cols + ['random_seed']
    combined_df = pd.merge(full_reopen_df[merge_cols + ['final infected fraction']],
                           full_close_df[merge_cols + ['final infected fraction']],
                           on=merge_cols,
                           validate='one_to_one',
                           how='inner',
                           suffixes=['_reopen', '_closed'])
    assert len(combined_df) == len(full_reopen_df) == len(full_close_df)
    combined_df['reopening_impact'] = (combined_df['final infected fraction_reopen'] -
                                       combined_df['final infected fraction_closed'])
    combined_df['total_additional_infections_from_reopening'] = combined_df['reopening_impact'] * combined_df['MSA_name'].map(lambda x:total_modeled_pops[x])
    # multiply by 10^5 to get incidence
    combined_df['reopening_impact'] = combined_df['reopening_impact'] * (10**5)
    print("Reopening impact quantifies cases per 100k")


    n_pois_in_cat = []
    for i in range(len(combined_df)):
        n_pois_in_cat.append(subcategory_counts[combined_df['MSA_name'].iloc[i]]
                             [combined_df['pretty_cat_names'].iloc[i]])
    combined_df['n_pois_in_cat'] = n_pois_in_cat
    combined_df['reopening_impact_per_poi'] = combined_df['reopening_impact'] / combined_df['n_pois_in_cat']

    

    if len(msa_names) == 1:
        print("Stats on mean additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index()))
        mean_impact = (combined_df[['pretty_cat_names', 'reopening_impact']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index())

        mean_impact_per_poi = (combined_df[['pretty_cat_names', 'reopening_impact_per_poi']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact_per_poi')[::-1].reset_index())
    else:
        # Want to make sure to weight each MSA equally, so have to take means twice.
        mean_impact = (combined_df[['MSA_name', 'pretty_cat_names', 'reopening_impact']]
                       .groupby(['pretty_cat_names', 'MSA_name'])
                       .mean()
                       .groupby('pretty_cat_names')
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index())

        mean_impact_per_poi = (combined_df[['MSA_name', 'pretty_cat_names', 'reopening_impact_per_poi']]
                       .groupby(['pretty_cat_names', 'MSA_name'])
                       .mean()
                       .groupby('pretty_cat_names')
                       .mean()
                       .sort_values(by='reopening_impact_per_poi')[::-1].reset_index())

    mean_poi_characteristics = poi_characteristics_df[['pretty_name', 'original_dwell_times', 'weighted_visits_over_area', 'density*dwell_time_factor', 'squared_visits_over_area', 'visits^2*dwell_time_factor/area']].groupby('pretty_name').mean().reset_index()
    compute_correlations = pd.merge(mean_impact_per_poi, mean_poi_characteristics, left_on='pretty_cat_names', right_on='pretty_name', validate='one_to_one', how='inner')
    assert len(compute_correlations) == len(mean_poi_characteristics)
    print("Pearson correlations between attributes")
    print(compute_correlations.corr(method='pearson')['reopening_impact_per_poi'])
    print("Spearman correlations between attributes")
    print(compute_correlations.corr(method='spearman')['reopening_impact_per_poi'])

    outlier_size = 1

    assert (combined_df['reopening_impact'] > 0).mean() > .8 # make sure we don't flip sign.
    fig = plt.figure(figsize=[12, 8])
    for i, poi_characteristic in enumerate(['original_dwell_times', 'weighted_visits_over_area']):
        ax = fig.add_subplot(2, 2, i + 1)
        sns.boxplot(y="pretty_name",
                x=poi_characteristic,
                data=poi_characteristics_df,
                order=list(mean_impact['pretty_cat_names']),
                ax=ax,
                fliersize=outlier_size)
        #ax.set_xscale('log')
        ax.set_ylabel("")
        if poi_characteristic == 'poi_areas':
            ax.set_xlabel("Area (sq feet)")
        elif poi_characteristic == 'original_dwell_times':
            ax.set_xlabel("Dwell time (minutes)")
            ax.set_xlim([0, 200])
        elif poi_characteristic == 'weighted_visits_over_area':
            ax.set_xlabel("Average visits per hour / sq ft")
            ax.set_xlim([1e-4, 1e-2])
        #ax.set_xlim([1e-8, 1e-5])
        ax.grid(alpha=0.5)


    ax = fig.add_subplot(2, 2, 3)
    sns.boxplot(y="pretty_cat_names",
                x="reopening_impact_per_poi",
                data=combined_df,
                order=mean_impact['pretty_cat_names'],
                ax=ax,
                whis=0,
                fliersize=outlier_size)
    ax.set_xscale('log')
    ax.set_ylabel("")
    ax.set_xlabel("Additional infections (per 100k), compared to not reopening (per POI)")
    ax.set_xlim([1e-2, 10])
    ax.grid(alpha=0.5)
    #ax.grid(alpha=.5)
    ax = fig.add_subplot(2, 2, 4)
    sns.boxplot(y="pretty_cat_names",
                x="reopening_impact",
                data=combined_df,
                order=mean_impact['pretty_cat_names'],
                ax=ax,
                whis=0,
                fliersize=outlier_size)
    ax.set_xlabel("Additional infections (per 100k), compared to not reopening")
    ax.set_xscale('log')
    ax.set_ylabel("")
    ax.set_xlim([10, 1e5])
    fig.suptitle(titlestring)
    plt.subplots_adjust(wspace=.6)
    ax.grid(alpha=0.5)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def plot_demographic_attributes_for_msa(msa_name):

    # Load geometry files. They're saved by states so just load the states we need.

    cbg_mapper = dataprocessor.CensusBlockGroups(base_directory='/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/census_block_group_shapefiles_by_state/',
                                             gdb_files=MSAS_TO_STATE_CBG_FILES[msa_name])
    msa_shapefile = gpd.read_file('/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/msa_shapefiles/tl_2017_us_cbsa/').to_crs(WGS_84_CRS)
    msa_shapefile['name_without_spaces'] = msa_shapefile['NAME'].map(
        lambda x:re.sub('[^0-9a-zA-Z]+', '_', x))

    # Load model with list of rich and poor CBGs which we're modeling. Don't actually fit it.
    data_kwargs = {'MSA_name':msa_name, 'nrows':None}
    model_kwargs = model_kwargs = {
        'min_datetime':datetime.datetime(2020, 3, 1, 0),
        'max_datetime':datetime.datetime(2020, 4, 11, 23),
        'cbgs_to_seed_in':None,
        'cbg_count_cutoff':10,
        'exogenous_model_kwargs':{
            'home_beta':0.01,
            'poi_psi':500,
            'p_sick_at_t0':0.0001,
            'just_compute_r0':False},
        'model_init_kwargs':{
            'starting_seed':0,
            'num_seeds':10,
            'approx_method':'poisson'},
        'simulation_kwargs':{
            'verbosity':24},
         'verbose':True,
        'cbg_day_prop_out':None,
        'include_cbg_prop_out':False,
        'filter_for_cbgs_in_msa':False,
        'return_model_without_fitting':True
    }
    fitted_model = fit_and_save_one_model(timestring='2020_04_01',
                                     model_kwargs=model_kwargs,
                                     data_kwargs=data_kwargs,
                                     train_test_partition=None)

    # Make plot.
    subplot_idx = 1
    fig = plt.figure(figsize=[20, 20])
    for attribute in ['median_household_income', 'p_black']:

        top_decile_cbgs = set(np.array(fitted_model.ALL_UNIQUE_CBGS)[
            fitted_model.cbg_idx_groups_to_track['%s_top_decile' % attribute]])

        bottom_decile_cbgs = set(np.array(fitted_model.ALL_UNIQUE_CBGS)[
            fitted_model.cbg_idx_groups_to_track['%s_bottom_decile' % attribute]])

        above_median_cbgs = set(np.array(fitted_model.ALL_UNIQUE_CBGS)[
            fitted_model.cbg_idx_groups_to_track['%s_above_median' % attribute]])

        below_median_cbgs = set(np.array(fitted_model.ALL_UNIQUE_CBGS)[
            fitted_model.cbg_idx_groups_to_track['%s_below_median' % attribute]])

        all_modeled_cbgs = set(np.array(fitted_model.ALL_UNIQUE_CBGS))

        cbg_mapper.block_group_d['id_to_match_to_safegraph_data'] = cbg_mapper.block_group_d['GEOID'].map(
            lambda x:x.split("US")[1]).astype(int)
        cbg_mapper.block_group_d['cbg_decile_category'] = (cbg_mapper.block_group_d['id_to_match_to_safegraph_data']
                                                    .map(lambda x:'above 90%' if x in top_decile_cbgs
                                                         else 'below 10%' if x in bottom_decile_cbgs
                                                         else 'middle deciles' if x in all_modeled_cbgs
                                                         else 'not in model'))
        cbg_mapper.block_group_d['cbg_median_category'] = (cbg_mapper.block_group_d['id_to_match_to_safegraph_data']
                                                    .map(lambda x:'above median' if x in above_median_cbgs
                                                         else 'below median' if x in below_median_cbgs
                                                         else 'other' if x in all_modeled_cbgs
                                                         else 'not in model'))

        print("Attribute: %s; MSA %s" % (attribute, msa_name))

        print(cbg_mapper.block_group_d[['cbg_decile_category', 'people_per_mile']]
          .groupby('cbg_decile_category').agg(['mean', 'size']))

        print(cbg_mapper.block_group_d[['cbg_median_category', 'people_per_mile']]
          .groupby('cbg_median_category').agg(['mean', 'size']))

        ax = fig.add_subplot(3, 3, subplot_idx)
        cbg_mapper.geometry_d.plot(
            column=cbg_mapper.block_group_d[attribute].values,
            cmap='Blues',
            linewidth=0.8,
            ax=ax)
        msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
        assert len(msa_boundary) == 1
        msa_boundary.boundary.plot(color='black', ax=ax)

        msa_border_padding = 0.1
        minx, maxx = msa_boundary.boundary.bounds[['minx', 'maxx']].iloc[0].to_list()
        ax.set_xlim([minx - msa_border_padding, maxx + msa_border_padding])

        miny, maxy = msa_boundary.boundary.bounds[['miny', 'maxy']].iloc[0].to_list()
        ax.set_ylim([miny - msa_border_padding, maxy + msa_border_padding])
        ax.set_title(attribute)
        subplot_idx += 1


        # plot deciles.
        ax = fig.add_subplot(3, 3, subplot_idx)

        cbg_mapper.geometry_d.plot(
            column=cbg_mapper.block_group_d['cbg_decile_category'].values,
            cmap='tab10',
            legend=True,
            linewidth=0.8, ax=ax)
        msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
        assert len(msa_boundary) == 1
        msa_boundary.boundary.plot(color='black', ax=ax)

        minx, maxx = msa_boundary.boundary.bounds[['minx', 'maxx']].iloc[0].to_list()
        ax.set_xlim([minx - msa_border_padding, maxx + msa_border_padding])

        miny, maxy = msa_boundary.boundary.bounds[['miny', 'maxy']].iloc[0].to_list()
        ax.set_ylim([miny - msa_border_padding, maxy + msa_border_padding])
        fig.tight_layout()
        ax.set_title(attribute)
        subplot_idx += 1

        # plot medians.
        ax = fig.add_subplot(3, 3, subplot_idx)

        cbg_mapper.geometry_d.plot(
            column=cbg_mapper.block_group_d['cbg_median_category'].values,
            cmap='tab10',
            legend=True,
            linewidth=0.8, ax=ax)
        msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
        assert len(msa_boundary) == 1
        msa_boundary.boundary.plot(color='black', ax=ax)

        minx, maxx = msa_boundary.boundary.bounds[['minx', 'maxx']].iloc[0].to_list()
        ax.set_xlim([minx - msa_border_padding, maxx + msa_border_padding])

        miny, maxy = msa_boundary.boundary.bounds[['miny', 'maxy']].iloc[0].to_list()
        ax.set_ylim([miny - msa_border_padding, maxy + msa_border_padding])
        fig.tight_layout()
        ax.set_title(attribute)
        subplot_idx += 1

    # plot density.
    ax = fig.add_subplot(3, 3, subplot_idx)
    cbg_mapper.geometry_d.plot(
        column=np.log10(cbg_mapper.block_group_d['people_per_mile'].values + 1),
        cmap='Blues',
        legend=True,
        linewidth=0.8, ax=ax)
    msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
    assert len(msa_boundary) == 1
    msa_boundary.boundary.plot(color='black', ax=ax)

    minx, maxx = msa_boundary.boundary.bounds[['minx', 'maxx']].iloc[0].to_list()
    ax.set_xlim([minx - msa_border_padding, maxx + msa_border_padding])

    miny, maxy = msa_boundary.boundary.bounds[['miny', 'maxy']].iloc[0].to_list()
    ax.set_ylim([miny - msa_border_padding, maxy + msa_border_padding])
    fig.tight_layout()
    ax.set_title('Log10 population density, people per mile')

    plt.savefig('%s_map.png' % msa_name, dpi=150)

def make_slir_plot_stratified_by_demographic_attribute(mdl, ax, attribute, median_or_decile,
                                                       slir_lines_to_plot=None):
    """
    Given a demographic attribute, plot SLIR curves for people above and below median
    if median_or_decile = median, or top and bottom decile, if median_or_decile = decile.
    """
    if slir_lines_to_plot is None:
        slir_lines_to_plot = ['L+I+R']
    assert attribute in ['p_black', 'p_white', 'median_household_income']

    if median_or_decile not in ['median', 'decile', 'above_median_in_own_county']:
        raise Exception("median_or_decile should be 'median' or 'decile' or 'above_median_in_own_county'")
    if median_or_decile == 'median':
        groups_to_plot = [f'{attribute}_above_median', f'{attribute}_below_median']
        title = 'above and below median for %s' % attribute
    elif median_or_decile == 'decile':
        groups_to_plot = [f'{attribute}_top_decile', f'{attribute}_bottom_decile']
        title = 'top and bottom decile for %s' % attribute
    elif median_or_decile == 'above_median_in_own_county':
        groups_to_plot = [f'{attribute}_above_median_in_own_county', f'{attribute}_below_median_in_own_county']
        title = 'above and below COUNTY median for %s' % attribute

    if attribute != 'p_black':
        groups_to_plot = groups_to_plot[::-1] # keep underserved population consistent. Should always be solid line (first line plotted)

    lines_to_return = plot_slir_over_time(
        mdl,
        ax,
        groups_to_plot=groups_to_plot,
        lines_to_plot=slir_lines_to_plot,
        title=title)
    return lines_to_return

def make_slir_race_ses_plot(mdl, path_to_save=None, title_string=None, slir_lines_to_plot=None):
    """
    Plot SLIR curves stratified by race and SES.
    Returns a dictionary which stores the values for each SLIR curve.
    """
    all_results = {}
    fig = plt.figure(figsize=[30, 20])
    subplot_idx = 1
    for demographic_attribute in ['p_black', 'p_white', 'median_household_income']:
        for median_or_decile in ['median', 'decile', 'above_median_in_own_county']:
            ax = fig.add_subplot(3, 3, subplot_idx)
            results = make_slir_plot_stratified_by_demographic_attribute(
                mdl=mdl,
                ax=ax,
                attribute=demographic_attribute,
                median_or_decile=median_or_decile,
                slir_lines_to_plot=slir_lines_to_plot)
            for k in results:
                assert k not in all_results
                all_results[k] = results[k]
            subplot_idx += 1
    if title_string is not None:
        fig.suptitle(title_string)
    if path_to_save is not None:
        fig.savefig(path_to_save)
    else:
        plt.show()
    return all_results

def make_ses_disparities_infection_ratio_plot_for_paper(all_ses_ratios, comparisons, demographic, filename):
    assert demographic in ['median_household_income', 'p_white']
    assert all([comparison in ['deciles', 'medians'] for comparison in comparisons])
    fig = plt.figure(figsize=[len(comparisons) * 6, 6])

    for ax_idx, comparison in enumerate(comparisons):
        ax = fig.add_subplot(1, len(comparisons), ax_idx + 1)
        rows_to_use = all_ses_ratios.loc[(all_ses_ratios['demo'] == demographic) & (all_ses_ratios['comparison'] == comparison)].copy()
        sns.boxplot(x='ratio',
            y='MSA',
            data=rows_to_use,
            color='lightblue',
            whis=0,
            fliersize=1)
        all_vals = rows_to_use['ratio'].values
        most_extreme_deviation = np.max(all_vals)
        ax.set_xlim([.4, 60])
        ax.set_xscale('log')
        ylimits = [-.5, len(set(rows_to_use['MSA'])) - .5]
        ax.plot([1, 1], ylimits, color='grey', linestyle='--')
        ax.set_ylim(ylimits)
        ax.set_ylabel("")
        if ax_idx == 1:
            ax.set_yticks([])


        if demographic == 'median_household_income':
            if comparison == 'deciles':
                ax.set_title("Disparities between CBGs\nin top and bottom income deciles", fontsize=14)
            else:
                ax.set_title("Disparities between CBGs\nabove and below median income", fontsize=14)
            ax.set_xlabel("Relative infection risk for lower-income CBGs", fontsize=14)
        elif demographic == 'p_white':
            if comparison == 'deciles':
                ax.set_title("Disparities between CBGs\nin top and bottom deciles for % white", fontsize=14)
            else:
                ax.set_title("Disparities between CBGs\nabove and below median for % white", fontsize=14)
            ax.set_xlabel("Relative infection risk for less white CBGs", fontsize=14)
        else:
            raise Exception("Invalid demographic variable")

        plt.xticks([0.5, 1, 2, 5, 10, 20], ['0.5x', '1x', '2x', '5x', '10x', '20x'])
        ax.tick_params(labelsize=14)
        plt.savefig(filename, bbox_inches='tight')


def plot_slir_over_time(mdl,
    ax,
    plot_logarithmic=True,
    timesteps_to_plot=None,
    groups_to_plot=None,
    lines_to_plot=None,
    title=None):
    """
    Plot SLIR fractions over time.
    """

    if groups_to_plot is None:
        groups_to_plot = ['all']
    history = copy.deepcopy(mdl.history)
    for group in history.keys():
        history[group]['L+I+R'] = history[group]['latent'] + history[group]['infected'] + history[group]['removed']

    if lines_to_plot is None:
        lines_to_plot = ['susceptible', 'latent', 'infected', 'removed']

    linestyles = ['-', '--', '-.', ':']
    colors = ['black', 'orange', 'blue', 'green', 'red']
    lines_to_return = {}

    for line_idx, k in enumerate(lines_to_plot):
        for group_idx, group in enumerate(groups_to_plot):
            total_population = history[group]['total_pop']
            time_in_days = np.arange(history[group][k].shape[1]) / 24.
            x = time_in_days
            y = (history[group][k].T / total_population).T
            assert y.shape[1] == x.shape[0]
            mean_Y, lower_CI_Y, upper_CI_Y = mean_and_CIs_of_timeseries_matrix(y)
            assert len(mean_Y) == len(x)

            color = colors[line_idx % len(colors)]
            linestyle = linestyles[group_idx % len(linestyles)]
            n_cbgs = history[group]['num_cbgs']
            if timesteps_to_plot is not None:
                x = x[:timesteps_to_plot]
                mean_Y = mean_Y[:timesteps_to_plot]

            ax.plot(x, mean_Y, label='%s, %s (n CBGs=%i)' % (k, group, n_cbgs), color=color, linestyle=linestyle)
            ax.fill_between(x, lower_CI_Y, upper_CI_Y, color=color, alpha=.2)

            if plot_logarithmic:
                ax.set_yscale('log')

            lines_to_return['%s, %s' % (k, group)] = mean_Y
    ax.legend() # Removed for now because we need to handle multiple labels
    logarithmic_string = ' (logarithmic)' if plot_logarithmic else ''
    ax.set_xlabel('Time (in days)')
    ax.set_ylabel("Fraction of population%s" % logarithmic_string)
    ax.set_xticks(range(math.ceil(max(time_in_days)) + 1))
    plt.xlim(0, math.ceil(max(time_in_days)))
    if plot_logarithmic:
        ax.set_ylim([1e-6, 1])
    else:
        ax.set_ylim([-.01, 1])
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=.5)
    return lines_to_return

def plot_new_cases_over_time(mdl, ax, agg_over_cbg=False, cumulative=True,
                             normalize_by_population_size=False, timesteps_to_plot=None):
    '''
    Plots the trend of new cases over time.
    mdl: a Model object that has completed a simulation
    ax: a matplotlib axis.
    agg_over_cbg: whether to plot a single line that sums the number of cases
                  over all CBGs, or to plot a line for each CBG
    cumulative: whether to plot the cumulative number of cases, or only the
                number of new cases at each timestep
    normalize_by_population_size: whether to normalize the line by the
                                  population size
    timesteps_to_plot: if not None, only plot this many timesteps.
    '''
    assert(len(mdl.new_cases_history) > 0)
    history = np.array(mdl.new_cases_history)  # time x CBG
    x = np.arange(len(history))/24. # time in days .

    if agg_over_cbg:
        y = np.sum(history, axis=1)  # sum over CBGs per time
        if cumulative:
            y = get_cumulative(y)
        if normalize_by_population_size:
            population_size = np.sum(mdl.CBG_SIZES)
            y = y / population_size
        if timesteps_to_plot is not None:
            ax.plot(x[:timesteps_to_plot], y[:timesteps_to_plot])
            ax.set_xlim([min(x), max(x)])
            ax.set_ylim([min(y), max(y)])
        else:
            ax.plot(x, y)

    else:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        linestyles = ['-', ':', '--', '-.']
        alphas = [0.5, 0.8, 0.5, 0.5]
        history = history.T  # CBG x time
        for i in range(mdl.N):
            y = history[i]
            if cumulative:
                y = get_cumulative(y)
            if normalize_by_population_size:
                population_size = mdl.CBG_SIZES[i]
                y = y / population_size
            idx = i % len(linestyles)
            if timesteps_to_plot is not None:
                x_to_plot, y_to_plot = x[:timesteps_to_plot], y[:timesteps_to_plot]
            else:
                x_to_plot, y_to_plot = x, y
            plt.plot(x_to_plot, y_to_plot, alpha=alphas[idx], linestyle=linestyles[idx],
                    color=colors[i], linewidth=4, label=i)
        ax.legend()
        ax.set_ylim(0, np.max(mdl.CBG_SIZES))

    ax.set_xlabel('Time (in days)')
    if cumulative:
        ylabel = 'Total number of cases'
    else:
        ylabel = 'Number of new cases'
    if normalize_by_population_size:
        ylabel += ' (% of population)'
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(math.ceil(max(x)) + 1))
    ax.set_xlim([0, math.ceil(max(x))])
    ax.grid(alpha=.5)

def plot_hourly_poi_visits(mdl, ax, timesteps_to_plot=None):
    total_poi_visits_by_timestep = mdl.POI_TIME_COUNTS.sum(axis=0)
    time_in_days = np.arange(len(total_poi_visits_by_timestep))/24. # time in days.

    if timesteps_to_plot is not None:
        ax.plot(time_in_days[:timesteps_to_plot], total_poi_visits_by_timestep[:timesteps_to_plot])
    else:
        ax.plot(time_in_days, total_poi_visits_by_timestep)
    ax.set_xticks(range(math.ceil(max(time_in_days)) + 1))
    ax.set_xlim([0, math.ceil(max(time_in_days))])
    ax.set_ylim([0, max(total_poi_visits_by_timestep)])
    ax.set_ylabel("Total visits to POIs")
    ax.set_xlabel("Time in days")
    ax.grid(alpha=.5)

def make_map_of_disease_spread(
    model,
    data_and_model_kwargs,
    lir_by_cbg_averaged_across_models,# average across top 10 models for consistency.
    cbg_mapper=None,
    filename=None):


    # Extract mapping data.
    msa_name = data_and_model_kwargs['data_kwargs']['MSA_name']
    if cbg_mapper is None:
        cbg_mapper = dataprocessor.CensusBlockGroups(base_directory='/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/census_block_group_shapefiles_by_state/',
                                             gdb_files=MSAS_TO_STATE_CBG_FILES[msa_name])
    msa_shapefile = gpd.read_file('/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/msa_shapefiles/tl_2017_us_cbsa/').to_crs(WGS_84_CRS)
    msa_shapefile['name_without_spaces'] = msa_shapefile['NAME'].map(
            lambda x:re.sub('[^0-9a-zA-Z]+', '_', x))
    msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
    assert len(msa_boundary) == 1

    # Extract final fraction in LIR by CBG.
    lir_df = pd.DataFrame({'cbg':model.ALL_UNIQUE_CBGS,
                           'n_in_LIR':lir_by_cbg_averaged_across_models,
                           'total_population':model.CBG_SIZES})
    lir_df['lir_per_capita'] = lir_df['n_in_LIR'] / lir_df['total_population']

    # Extract where people are going out.
    ipf_output = pickle.load(open(get_ipf_filename(msa_name,
        data_and_model_kwargs['model_kwargs']['min_datetime'],
        data_and_model_kwargs['model_kwargs']['max_datetime'],
        clip_visits=data_and_model_kwargs['model_kwargs']['poi_attributes_to_clip']['clip_visits']), 'rb'))
    timesteps_before_end = range(-168, 0)
    total_n_out_over_week = 0
    for t in timesteps_before_end:
        if t % 24 == 0:
            print("Processing IPF output timestep %i" % t)
        poi_cbg_array = ipf_output[t].toarray()
        assert poi_cbg_array.shape[1] == len(model.CBG_SIZES)
        n_out = poi_cbg_array.sum(axis=0)
        total_n_out_over_week = total_n_out_over_week + n_out
    total_n_out_over_week_per_capita = total_n_out_over_week/model.CBG_SIZES
    lir_df['total_n_out_over_week_per_capita'] = total_n_out_over_week_per_capita

    # Extract true social distancing numbers.
    sdm = helper.load_social_distancing_metrics(
    helper.list_datetimes_in_range(
        data_and_model_kwargs['model_kwargs']['max_datetime'] - datetime.timedelta(days=6),
        data_and_model_kwargs['model_kwargs']['max_datetime']))
    cbg_day_prop_out = helper.compute_cbg_day_prop_out(sdm)
    cols_to_average = [a for a in cbg_day_prop_out.columns if a != 'census_block_group']
    cbg_day_prop_out['mean_frac_out_from_sdm'] = cbg_day_prop_out[cols_to_average].values.mean(axis=1)

    lir_df = pd.merge(lir_df, cbg_day_prop_out[['mean_frac_out_from_sdm', 'census_block_group']],
        how='inner', left_on='cbg', right_on='census_block_group', validate='one_to_one')

    # match to geometry D.
    geometry_d = cbg_mapper.geometry_d.copy()
    geometry_d['cbg'] = geometry_d['GEOID_Data'].map(lambda x:x.split("US")[1]).astype(int)
    geometry_d = pd.merge(geometry_d,
                          lir_df,
                          on='cbg',
                          how='inner',
                          validate='one_to_one')


    acs_d = helper.load_and_reconcile_multiple_acs_data()[['p_black_2017_5YR',  'p_white_2017_5YR',
    'median_household_income_2017_5YR', 'census_block_group', 'people_per_mile_hybrid']]
    geometry_d = pd.merge(geometry_d,
        acs_d,
        left_on='cbg',
        right_on='census_block_group',
        how='inner',
        validate='one_to_one')
    print("Matched %i CBGs out of %i in SafeGraph data" % (len(geometry_d), len(lir_df)))

    all_cols_to_plot = ['lir_per_capita', 'people_per_mile_hybrid', 'total_n_out_over_week_per_capita', 'median_household_income_2017_5YR', 'p_black_2017_5YR', 'mean_frac_out_from_sdm']
    geometry_d = geometry_d.dropna(subset=all_cols_to_plot)
    print("After dropping NAs, %i rows" % len(geometry_d))

    print("Spearman correlation between all pairs of variables (this is assessed over all CBGs, not just those in the MSA)")
    correlation_df = []
    for c in all_cols_to_plot:
        for c2 in all_cols_to_plot:
            if c > c2:
                r, p = spearmanr(geometry_d[c], geometry_d[c2])
                correlation_df.append({'var1':c, 'var2':c2, 'spearman_r':r, 'spearman_p':p})
    print(pd.DataFrame(correlation_df).sort_values(by='spearman_p')[['var1', 'var2', 'spearman_r', 'spearman_p']])


    # Filter for CBGs in MSA (just for plotting).
    nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(msa_name)
    msa_cbgs = set(msa_cbgs)
    geometry_d = geometry_d.loc[geometry_d['cbg'].map(lambda x:x in msa_cbgs)]
    print("After filtering for CBGs in MSA, %i CBGs" % (len(geometry_d)))

    fig = plt.figure(figsize=[15, 10])




    for subplot_idx, col_to_plot in enumerate(all_cols_to_plot):
        ax = fig.add_subplot(2, 3, subplot_idx + 1)
        if col_to_plot == 'lir_per_capita':
            #vmin = -3
            #vmax = 0
            ax.set_title("Infected fraction\n(percentile among CBGs)", fontsize=16)
        elif col_to_plot == 'people_per_mile_hybrid':
            #vmin = 2
            #vmax = 4
            ax.set_title('Population density\n(percentile among CBGs)', fontsize=16)
        elif col_to_plot == 'p_black_2017_5YR':
            #vmin = 0
            #vmax = 1
            ax.set_title("Proportion black\n(percentile among CBGs)", fontsize=16)
            #take_log = False
        elif col_to_plot == 'median_household_income_2017_5YR':
            #vmin = None
            #vmax = None
            ax.set_title("Median household income\n(percentile among CBGs)", fontsize=16)
        elif col_to_plot == 'total_n_out_over_week_per_capita':
            ax.set_title("# hours outside of home\nin final week\n(percentile among CBGs)", fontsize=16)
        elif col_to_plot == 'mean_frac_out_from_sdm':
            ax.set_title("Average fraction outside home\nin final week (from SDM)\n(percentile among CBGs)", fontsize=16)

            #take_log = False
        geometry_d['percentile_scored'] = 100.*rankdata(geometry_d[col_to_plot].values) / len(geometry_d[col_to_plot].values)
        #geometry_d['decile'] = np.array([10 * math.ceil(x / 10.) for x in geometry_d['percentile_scored']])
        geometry_d.plot(
                column=geometry_d['percentile_scored'].values,
                cmap='Reds',
                linewidth=0.8,
                ax=ax,
                vmin=0,
                vmax=100,
                legend=True)
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
        assert len(msa_boundary) == 1
        msa_boundary.boundary.plot(color='black', ax=ax)
        msa_border_padding = 0

        minx, maxx = msa_boundary.boundary.bounds[['minx', 'maxx']].iloc[0].to_list()
        miny, maxy = msa_boundary.boundary.bounds[['miny', 'maxy']].iloc[0].to_list()
        ax.set_xlim([minx - msa_border_padding , maxx + msa_border_padding])
        ax.set_ylim([miny - msa_border_padding, maxy + msa_border_padding])
        ax.set_xticks([])
        ax.set_yticks([])
        #cbar = plt.cm.ScalarMappable(cmap='RdBu')
        #ax.get_figure().colorbar(n_cmap, ax=ax, orientation='horizontal')
    fig.suptitle(MSAS_TO_PRETTY_NAMES[msa_name], fontsize=18)
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    return cbg_mapper, geometry_d, msa_boundary

def get_list_of_poi_subcategories_with_most_visits(n_poi_categories):
    """
    Remove blacklisted categories and return n_poi_categories subcategories with the most visits in "normal times" (Jan 2019 - Feb 2020)
    """
    normal_times = helper.list_datetimes_in_range(datetime.datetime(2019, 1, 1),
                                              datetime.datetime(2020, 2, 29))
    normal_time_cols = ['%i.%i.%i' % (a.year, a.month, a.day) for a in normal_times]
    must_have_cols = normal_time_cols + ['sub_category', 'top_category']
    d = helper.load_multiple_chunks(range(5), cols=must_have_cols)
    d['visits_in_normal_times'] = d[normal_time_cols].sum(axis=1)
    assert all([a in d['sub_category'].values for a in SUBCATEGORY_BLACKLIST])
    assert((d.groupby('sub_category')['top_category'].nunique().values == 1).all()) # Make sure that each sub_category only maps to one top category (and so it's safe to just look at sub categories).
    d = d.loc[d['sub_category'].map(lambda x:x not in SUBCATEGORY_BLACKLIST)]
    grouped_d = d.groupby('sub_category')['visits_in_normal_times'].sum().sort_values()[::-1].iloc[:n_poi_categories]
    print("Returning the list of %i POI subcategories with the most visits, collectively accounting for percentage %2.1f%% of visits" %
        (n_poi_categories, 100*grouped_d.values.sum()/d['visits_in_normal_times'].sum()))
    return list(grouped_d.index)


def plot_reopening_effect_by_poi_category_with_disparate_impact(intervention_df, medians_or_deciles, cats_to_plot, filename=None):
    print("Analyzing results for rich and poor %s" % medians_or_deciles)
    assert medians_or_deciles in ['medians', 'deciles']
    intervention_df = intervention_df.copy()
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x in cats_to_plot)]
    intervention_df['mean_final_infected_fraction'] = intervention_df['final infected fraction'].map(lambda x:np.mean(x))
    assert len([a for a in SUBCATEGORY_BLACKLIST if a in intervention_df['counterfactual_sub_category'].values]) == 0
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x not in SUBCATEGORY_BLACKLIST)]
    intervention_df['pretty_cat_names'] = intervention_df['counterfactual_sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
    full_reopen_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 1]
    full_close_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 0]

    assert set(full_close_df.groupby(['MSA_name', 'model_fit_rank_for_msa'])['final infected fraction'].size().values) == set([len(set(intervention_df['counterfactual_sub_category']))])
    assert set(full_close_df.groupby(['MSA_name', 'model_fit_rank_for_msa'])['mean_final_infected_fraction'].nunique().values) == set([1])

    full_reopen_df = full_reopen_df[['timestring', 'model_fit_rank_for_msa', 'pretty_cat_names', 'MSA_name']]
    full_close_df = full_close_df[['timestring', 'model_fit_rank_for_msa', 'MSA_name']]
    full_close_df = full_close_df.drop_duplicates(subset=['model_fit_rank_for_msa', 'MSA_name']) # should be all the same.
    print("%i models to load for closed df; %i models to load for opened interventions" % (len(full_close_df), len(full_reopen_df)))

    if medians_or_deciles == 'deciles':
        top_group = 'median_household_income_top_decile'
        bottom_group = 'median_household_income_bottom_decile'
    else:
        top_group = 'median_household_income_above_median'
        bottom_group = 'median_household_income_below_median'

    for setting in ['open', 'closed']:
        if setting == 'open':
            df_to_use = full_reopen_df
        else:
            df_to_use = full_close_df

        results_by_group = {top_group:[], bottom_group:[]}
        for i in range(len(df_to_use)):
            if i % 10 == 0:
                print(i)
            timestring = df_to_use.iloc[i]['timestring']
            mdl, _, _, _, _ = load_model_and_data_from_timestring(
                timestring,
                load_fast_results_only=False,
                load_full_model=True)
            for g in [top_group, bottom_group]:
                n_group = mdl.history[g]['total_pop']
                n_LIR = (mdl.history[g]['latent'] + mdl.history[g]['infected'] + mdl.history[g]['removed'])[:, -1]
                results_by_group[g].append(n_LIR/n_group)

        if setting == 'open':
            full_reopen_df['top_group_lir_if_opened'] = results_by_group[top_group]
            full_reopen_df['bottom_group_lir_if_opened'] = results_by_group[bottom_group]
            full_reopen_df = unpack_random_seeds(full_reopen_df,
                cols_of_interest=['top_group_lir_if_opened', 'bottom_group_lir_if_opened'],
                cols_to_keep=['model_fit_rank_for_msa', 'pretty_cat_names', 'MSA_name'])
        else:
            full_close_df['top_group_lir_if_closed'] = results_by_group[top_group]
            full_close_df['bottom_group_lir_if_closed'] = results_by_group[bottom_group]
            full_close_df = unpack_random_seeds(full_close_df,
                cols_of_interest=['top_group_lir_if_closed', 'bottom_group_lir_if_closed'],
                cols_to_keep=['model_fit_rank_for_msa', 'MSA_name'])

    combined_df = pd.merge(full_reopen_df[['pretty_cat_names', 'model_fit_rank_for_msa', 'random_seed', 'MSA_name', 'top_group_lir_if_opened', 'bottom_group_lir_if_opened']],
                           full_close_df[['model_fit_rank_for_msa', 'random_seed', 'MSA_name', 'top_group_lir_if_closed', 'bottom_group_lir_if_closed']],
                           on=['model_fit_rank_for_msa', 'random_seed', 'MSA_name'],
                           validate='many_to_one',
                           how='inner')

    for group in ['top', 'bottom']:
        combined_df['impact_of_reopening_%s' % group] = (combined_df['%s_group_lir_if_opened' % group] - combined_df['%s_group_lir_if_closed' % group]) * 10**5
        combined_df['relative_impact_of_reopening_%s' % group] = combined_df['impact_of_reopening_%s' % group]/(combined_df['%s_group_lir_if_closed' % group] * 10**5)


    combined_df['impact_worse_for_bottom'] = combined_df['impact_of_reopening_bottom'] > combined_df['impact_of_reopening_top']
    combined_df['relative_impact_worse_for_bottom'] = combined_df['relative_impact_of_reopening_bottom'] > combined_df['relative_impact_of_reopening_top']

    for comparison in ['impact_worse_for_bottom', 'relative_impact_worse_for_bottom']:
        print("\n\n*******************%s" % comparison)
        print("Overall, %i/%i models (proportion %2.3f) forecast %s" % (combined_df[comparison].sum(),
            len(combined_df), combined_df[comparison].mean(), comparison))
        for k in ['MSA_name', 'pretty_cat_names']:
            worse_for_poor = combined_df[[comparison, k]].groupby(k).agg(['sum', 'size', 'mean']).reset_index()
            worse_for_poor.columns = [k, 'sum', 'size', 'mean']
            print(worse_for_poor.sort_values(by='mean')[::-1])


    print("Warning: in fraction %2.3f of cases, reopening has POSITIVE impact on bottom group; mean in these cases is %2.5f" %
      ((combined_df['impact_of_reopening_bottom'] < 0).mean(),
       combined_df.loc[combined_df['impact_of_reopening_bottom'] < 0, 'impact_of_reopening_bottom'].mean()))
    print("Warning: in fraction %2.3f of cases, reopening has POSITIVE impact on top; mean in these cases is %2.5f" %
      ((combined_df['impact_of_reopening_top'] < 0).mean(),
       combined_df.loc[combined_df['impact_of_reopening_top'] < 0, 'impact_of_reopening_top'].mean()))

    msa_order = sorted(list(set(combined_df['MSA_name'])))

    order_to_display_cats = (combined_df[['relative_impact_of_reopening_bottom',
                                       'relative_impact_of_reopening_top',
                                        'impact_of_reopening_bottom',
                                         'impact_of_reopening_top',
                                       'pretty_cat_names']].groupby('pretty_cat_names')
         .mean()
         .reset_index()
        .sort_values(by='impact_of_reopening_bottom'))


    for k in order_to_display_cats.columns:
        for k2 in order_to_display_cats.columns:
            if k == 'pretty_cat_names' or k2 == 'pretty_cat_names':
                continue
            if k < k2:
                sorted_by_k = list(order_to_display_cats.sort_values(by=k)[::-1]['pretty_cat_names'].iloc[:5])
                sorted_by_k2 = list(order_to_display_cats.sort_values(by=k2)[::-1]['pretty_cat_names'].iloc[:5])

                print('%-30s %-30s spearman r=%2.5f; %i/5 top cats are same' % (k, k2, spearmanr(order_to_display_cats[k], order_to_display_cats[k2])[0],
                    len([a for a in sorted_by_k if a in sorted_by_k2])))

    order_to_display_cats = list(order_to_display_cats['pretty_cat_names'].values)

    fig = plt.figure(figsize=[16, 10])
    subplot_idx = 1
    for msa in msa_order:

        df_to_plot_for_msa = {'pretty_cat_names':[],
            'bottom_group_mean':[], 'bottom_group_upper_CI':[], 'bottom_group_lower_CI':[],
            'top_group_mean':[], 'top_group_upper_CI':[], 'top_group_lower_CI':[]}
        grouped_by_cat = combined_df.loc[combined_df['MSA_name'] == msa].groupby('pretty_cat_names')
        for pretty_cat_name in order_to_display_cats:
            small_df = grouped_by_cat.get_group(pretty_cat_name)
            df_to_plot_for_msa['pretty_cat_names'].append(pretty_cat_name)
            for top_or_bottom in ['top', 'bottom']:
                df_to_plot_for_msa['%s_group_mean' % top_or_bottom].append(small_df['impact_of_reopening_%s' % top_or_bottom].mean())
                df_to_plot_for_msa['%s_group_lower_CI' % top_or_bottom].append(max(1, scoreatpercentile(small_df['impact_of_reopening_%s' % top_or_bottom], 2.5)))
                df_to_plot_for_msa['%s_group_upper_CI' % top_or_bottom].append(max(1, scoreatpercentile(small_df['impact_of_reopening_%s' % top_or_bottom], 97.5)))
        df_to_plot_for_msa = pd.DataFrame(df_to_plot_for_msa)


        # also save figures individually.
        individual_fig = plt.figure(figsize=[6, 5])
        individual_ax = individual_fig.add_subplot(1, 1, 1)

        multi_ax = fig.add_subplot(2, 5, subplot_idx)
        yvals = range(len(df_to_plot_for_msa))

        for fig_to_make in ['all cities', 'individual cities']:
            if fig_to_make == 'all cities':
                ax = multi_ax
            else:
                ax = individual_ax
            # poor
            ax.plot(df_to_plot_for_msa['bottom_group_mean'].values,
                    yvals,
                    label='Below median\nincome' if medians_or_deciles == 'medians' else 'Bottom decile\nincome', color='darkorchid')
            ax.fill_betweenx(y=yvals,
                    x1=df_to_plot_for_msa['bottom_group_lower_CI'].values,
                    x2=df_to_plot_for_msa['bottom_group_upper_CI'].values, color='darkorchid', alpha=.2)

            # rich
            ax.plot(df_to_plot_for_msa['top_group_mean'].values,
                    yvals,
                    label='Above median\nincome' if medians_or_deciles == 'medians' else 'Top decile\nincome', color='darkgoldenrod')
            ax.fill_betweenx(y=yvals,
                    x1=df_to_plot_for_msa['top_group_lower_CI'].values,
                    x2=df_to_plot_for_msa['top_group_upper_CI'].values, color='darkgoldenrod', alpha=.2)

            if subplot_idx == 1 or subplot_idx == 6 or fig_to_make == 'individual cities':
                ax.set_yticks(yvals)
                ax.set_yticklabels(list(df_to_plot_for_msa['pretty_cat_names']))
            else:
                ax.set_yticks(yvals)
                ax.set_yticklabels([])
            ax.set_xscale('log')
            if subplot_idx == 1 or fig_to_make == 'individual cities':
                ax.legend()
            ax.set_title(MSAS_TO_PRETTY_NAMES[msa])
            ax.grid(alpha=.3)
            ax.set_xlim([10, 1e5])
            ax.set_ylim([min(yvals), max(yvals)])
            if subplot_idx > 5 or fig_to_make == 'individual cities':
                ax.set_xlabel("Additional infections (per 100k)\ncompared to not reopening")
            if fig_to_make == 'individual cities' and filename is not None:
                individual_fig.tight_layout()
                assert '.pdf' in filename
                individual_fig.savefig(filename.replace('.pdf', '_%s.pdf' % msa))
        subplot_idx += 1
    fig.subplots_adjust(hspace=.05)
    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
    plt.show()


def plot_min_and_max_of_multiple_models(timestrings, ax, label, color, min_datetime=None, alpha=0.5):
    curves = []
    hours = None
    for ts in timestrings:
        _, kwargs, _, model_results, _ = load_model_and_data_from_timestring(ts, load_fast_results_only=False,
                                                                             load_full_model=False)

        new_hours = [kwargs['model_kwargs']['min_datetime'] + datetime.timedelta(hours=a)
                 for a in range(model_results['history']['all']['latent'].shape[1])]
        if min_datetime is not None:
            hours_above_datetime = np.array(new_hours) >= min_datetime
            new_hours = np.array(new_hours)[hours_above_datetime]
        if hours is None:
            hours = new_hours
        else:
            assert list(hours) == list(new_hours) # make sure hours stays unchanged between timestrings.
        frac_in_lir = ((model_results['history']['all']['latent'] +
                       model_results['history']['all']['infected'] +
                       model_results['history']['all']['removed']).mean(axis=0)/
                       model_results['history']['all']['total_pop'])
        if min_datetime is not None:
            frac_in_lir = frac_in_lir[hours_above_datetime]
        assert len(frac_in_lir) == len(hours)
        curves.append(frac_in_lir)
    curves = np.array(curves)
    mean = np.mean(curves, axis=0)
    lower_bound = np.min(curves, axis=0)
    upper_bound = np.max(curves, axis=0)
    ax.plot_date(hours, mean, linestyle='-', label=label, color=color)
    ax.fill_between(hours, lower_bound, upper_bound, alpha=alpha, color=color)
    ax.set_xlim([min(hours), max(hours)])
    #ax.set_ylim([min(lower_bound), max(upper_bound)

def make_counterfactual_line_plots(counterfactual_df, msa, ax, mode,
                                   cmap_str='viridis', y_lim=None):
    assert mode in {'degree', 'shift-later', 'shift-earlier', 'shift'}

    colors = list(cm.get_cmap(cmap_str, 7).colors)
    colors.reverse()
    if mode == 'degree':
        values = [0, .25, .5, np.nan]  # put highest curve first, so that legend order is correct
        param_name = 'counterfactual_distancing_degree'
        subtitle = 'What if we had only socially distanced x%?'
        colors = colors[3:]
    elif mode == 'shift-later':
        values = [14, 7, 3, np.nan]
        param_name = 'counterfactual_shift_in_days'
        subtitle = 'What if we had started socially distancing x days later?'
    elif mode == 'shift-earlier':
        values = [np.nan, -3, -7, -14]
        param_name = 'counterfactual_shift_in_days'
        subtitle = 'What if we had started socially distancing x days earlier?'
        colors = colors[3:]  # so that true curve maintains the same color
    else:
        values = [14, 7, 3, np.nan, -3, -7, -14]
        param_name = 'counterfactual_shift_in_days'
        subtitle = 'What if we had started socially distancing earlier or later?'

    msa_df = counterfactual_df[counterfactual_df['MSA_name'] == msa]
    color_idx = 0
    for i, val in enumerate(values):
        if np.isnan(val):
            # plot baseline models for comparison
            timestrings = msa_df.counterfactual_baseline_model.unique()
            label = '100% (true)' if mode == 'degree' else '0 days (true)'
            plot_min_and_max_of_multiple_models(timestrings, ax, label, colors[i])
        else:
            # plot the models from this experiment
            msa_val_df = msa_df[msa_df[param_name] == val]
            timestrings = msa_val_df.timestring.values
            if mode == 'degree':
                label = '%d%%' % (val * 100)
            else:  # mode is some kind of shift
                postfix = 'earlier' if val < 0 else 'later'
                label = '%d days %s' % (abs(val), postfix)  # take abs value in case val is negative
            plot_min_and_max_of_multiple_models(timestrings, ax, label, colors[i])

    ax.legend(loc='upper left', fontsize=12)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Fraction of population infected', fontsize=14)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(alpha=.5)
    ax.tick_params(labelsize=12)
    ax.set_title(subtitle, fontsize=14)



def make_status_plots_for_each_simulation_timestep(mdl,
                        data_and_model_kwargs,
                         timesteps):

    """
    For each timestep in timesteps, this makes several plots -
    1. a map of where the disease is
    2. number of total cases
    3. SLIR fractions
    4. Hourly activity.

    Images can be linked together using a command like

    ~/Desktop/ffmpeg -framerate 3 -pattern_type glob -i '*.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

    Note: three of the four plots are put into their own method; the map plot code is in this function.
    May be worth pulling out at some point, but wanted to avoid duplicate work between simulation timesteps.
    """
    raise Exception("This method is deprecated as is - not sure plotting methods will work")


    for timestep in timesteps:
        fig = plt.figure(figsize=[18, 12])
        #plt.subplots(1, 2, figsize=[12, 6])
        fig.suptitle("Simulation hour %i\nCumulative number of cases %i" % (timestep, cumulative_cases[timestep, cbg_idxs].sum()))
        ax = fig.add_subplot(2, 2, 1)
        make_map_of_disease_spread(ax, mdl, data_and_model_kwargs) # this does not make a time-varying map at present.

        ax = fig.add_subplot(2, 2, 2)
        plot_new_cases_over_time(mdl, ax, agg_over_cbg=True, cumulative=True, timesteps_to_plot=timestep)

        ax = fig.add_subplot(2, 2, 3)
        plot_slir_over_time(mdl, ax, timesteps_to_plot=timestep)

        ax = fig.add_subplot(2, 2, 4)
        plot_hourly_poi_visits(mdl, ax, timesteps_to_plot=timestep)

        fig.savefig('test_figs/timestep_%05d.png' % timestep, dpi=150)
        plt.show()

def plot_model_fit_from_model_and_kwargs(ax,
                                         mdl_kwargs,
                                         data_kwargs,
                                         model=None,
                                         train_test_partition=None,
                                         model_results=None,
                                         plotting_kwargs=None):
    msa_name = data_kwargs['MSA_name']
    nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)
    min_datetime = mdl_kwargs['min_datetime']
    if plotting_kwargs is None:
        plotting_kwargs = {}  # could include options like plot_mode, plot_log, etc.
    if 'title' not in plotting_kwargs:
        plotting_kwargs['title'] = MSAS_TO_PRETTY_NAMES[msa_name]
    if 'make_plot' not in plotting_kwargs:
        plotting_kwargs['make_plot'] = True
    ret_val = compare_model_vs_real_num_cases(nyt_outcomes, min_datetime,
                                    model=model,
                                    model_results=model_results,
                                    ax=ax,
                                    **plotting_kwargs)
    if train_test_partition is not None and plotting_kwargs['make_plot']:
        ax.plot_date([train_test_partition, train_test_partition], ax.get_ylim(), color='black', linestyle='-')
    return ret_val

#########################################################
# Functions to evaluate model fit
#########################################################

def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    '''
    Matches the MSA name from our annotated SafeGraph data to the
    MSA name in the external datasource in helper.MSA_COUNTY_MAPPING
    '''
    msa_pieces = msa_name.split('_')
    query_states = set()
    i = len(msa_pieces) - 1
    while True:
        piece = msa_pieces[i]
        if len(piece) == 2 and piece.upper() == piece:
            query_states.add(piece)
            i -= 1
        else:
            break
    query_cities = set(msa_pieces[:i+1])

    for msa in acs_msas:
        if ', ' in msa:
            city_string, state_string = msa.split(', ')
            states = set(state_string.split('-'))
            if states == query_states:
                cities = city_string.split('-')
                overlap = set(cities).intersection(query_cities)
                if len(overlap) > 0:  # same states and at least one city matched
                    return msa
    return None

def get_fips_codes_from_state_and_county_fp(state_vec, county_vec):
    fips_codes = []
    for state, county in zip(state_vec, county_vec):
        state = str(state)
        if len(state) == 1:
            state = '0' + state
        county = str(county)
        if len(county) == 1:
            county = '00' + county
        elif len(county) == 2:
            county = '0' + county
        fips_codes.append(np.int64(state + county))
    return fips_codes

def get_nyt_outcomes_over_counties(counties=None):
    PATH_TO_NYT_DATA = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/nytimes_coronavirus_data/covid-19-data/us-counties.csv'
    outcomes = pd.read_csv(PATH_TO_NYT_DATA)
    if counties is not None:
        outcomes = outcomes[outcomes['fips'].isin(counties)]
    return outcomes

def get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes):
    date_groups = nyt_outcomes.groupby('date').indices
    dates = sorted(date_groups.keys())
    datetimes = []
    total_cases = []
    total_deaths = []
    for date in dates:
        year, month, day = date.split('-')
        curr_datetime = datetime.datetime(int(year), int(month), int(day))
        if len(datetimes) > 0:
            assert(curr_datetime > datetimes[-1])
        datetimes.append(curr_datetime)
        rows = nyt_outcomes.iloc[date_groups[date]]
        total_cases.append(np.sum(rows['cases'].to_numpy()))
        total_deaths.append(np.sum(rows['deaths'].to_numpy()))
    return datetimes, np.array(total_cases), np.array(total_deaths)

def find_model_and_real_overlap_for_eval(real_dates, real_cases, mdl_hours, mdl_cases,
                                         compare_start_time=None, compare_end_time=None):
    overlap = set(real_dates).intersection(set(mdl_hours))
    if compare_start_time is None:
        compare_start_time = min(overlap)
    if compare_end_time is None:
        compare_end_time = max(overlap)
    comparable_period = helper.list_hours_in_range(compare_start_time, compare_end_time)
    overlap = sorted(overlap.intersection(set(comparable_period)))
    real_date2case = dict(zip(real_dates, real_cases))
    mdl_date2case = dict(zip(mdl_hours, mdl_cases.T)) #mdl_cases has an extra random_seed first dim
    real_vec = []
    mdl_mat = np.zeros((len(mdl_cases), len(overlap)))  # num_seed x num_time
    for idx, date in enumerate(overlap):
        real_vec.append(real_date2case[date])
        mdl_mat[:, idx] = mdl_date2case[date]
    return np.array(real_vec), mdl_mat, overlap[0], overlap[-1]

def get_variables_for_evaluating_msa_model(msa_name, verbose=False):
    acs_data = pd.read_csv(PATH_TO_5_YEAR_ACS_DATA)
    acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
    msa_match = match_msa_name_to_msas_in_acs_data(msa_name, acs_msas)
    if verbose: print('Found MSA %s in ACS 5-year data' % msa_match)

    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['id_to_match_to_safegraph_data'] = msa_data['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
    msa_cbgs = msa_data['id_to_match_to_safegraph_data'].values
    msa_data['fips'] = get_fips_codes_from_state_and_county_fp(msa_data.STATEFP, msa_data.COUNTYFP)
    msa_counties = list(set(msa_data['fips'].values))
    if verbose:
        print('Found %d counties and %d CBGs in MSA' % (len(msa_counties), len(msa_cbgs)))

    nyt_outcomes = get_nyt_outcomes_over_counties(msa_counties)
    nyt_counties = set(nyt_outcomes.fips.unique())
    nyt_cbgs = msa_data[msa_data['fips'].isin(nyt_counties)]['id_to_match_to_safegraph_data'].values
    if verbose:
        print('Found NYT data matching %d counties and %d CBGs' % (len(nyt_counties), len(nyt_cbgs)))
    return nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs

def compare_model_vs_real_num_cases(nyt_outcomes,
                                    mdl_start_time,
                                    compare_start_time=None,
                                    compare_end_time=None,
                                    model=None,
                                    model_results=None,
                                    mdl_prediction=None,
                                    projected_hrs=None,
                                    detection_rate=.10, #.12
                                    detection_lag=7, #9
                                    death_rate=.0066, #.005
                                    death_lag=18, #14
                                    verbose=False,
                                    make_plot=False,
                                    ax=None,
                                    title=None,
                                    plot_log=False,
                                    plot_mode='cases',
                                    plot_errorbars=False,
                                    plot_real_data=True,
                                    plot_daily_not_cumulative=False,
                                    only_plot_intersection=True,
                                    model_line_label=None,
                                    true_line_label=None,
                                    x_interval=None,
                                    smooth_daily_cases_when_plotting=False,
                                    title_fontsize=20,
                                    legend_fontsize=16,
                                    tick_label_fontsize=16,
                                    marker_size=2,
                                    plot_legend=True,
                                    real_data_color='black',
                                    model_color='darkorchid',
                                    xticks=None,
                                    x_range=None,
                                    y_range=None,
                                    only_two_yticks=False,
                                    return_mdl_pred_and_hours=False):
    assert plot_daily_not_cumulative in [True, False]
    if model is not None:
        cbgs_to_idxs = model.CBGS_TO_IDXS
        history = model.history
        assert('nyt' in history)
        assert model_results is None
        assert mdl_prediction is None
        assert projected_hrs is None
    elif model_results is not None:
        cbgs_to_idxs = model_results['CBGS_TO_IDXS']
        history = model_results['history']
        assert('nyt' in history)
        assert mdl_prediction is None
        assert projected_hrs is None
    else:
        assert mdl_prediction is not None
        assert projected_hrs is not None


    real_dates, real_cases, real_deaths = get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes)
    score_dict = {}

    if mdl_prediction is not None:
        mdl_prediction_provided = True
    else:
        mdl_prediction_provided = False

    if not mdl_prediction_provided:
        # align cases with datetimes
        mdl_IR = (history['nyt']['infected'] + history['nyt']['removed']) # should think of this as a cumulative count because once you enter the removed state, you never leave. So mdl_cases is the number of people who have _ever_ been infectious or removed (ie, in states I or R).
        num_hours = mdl_IR.shape[1]
        mdl_end_time = mdl_start_time + datetime.timedelta(hours=num_hours-1)
        mdl_hours = helper.list_hours_in_range(mdl_start_time, mdl_end_time)
        mdl_dates = helper.list_datetimes_in_range(mdl_start_time, mdl_end_time)
        assert(mdl_start_time < mdl_end_time)

    modes = ['cases', 'deaths']

    for mode in modes:

        if mode == 'cases':
            real_data = real_cases
        else:
            real_data = real_deaths

        if not mdl_prediction_provided:
            if mode == 'cases':
                MRE_thresh = 50 # don't evaluate MRE on very small numbers -- too noisy
                if 'new_confirmed_cases' in history['nyt']:  # modeled confirmed cases during simulation
                    mdl_hourly_new_cases = history['nyt']['new_confirmed_cases']
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
                else:  # need to extrapolate confirmed cases from I+R curve
                    mdl_prediction = mdl_IR * detection_rate
                    projected_hrs = [hr + datetime.timedelta(days=detection_lag) for hr in mdl_hours]
            else:
                MRE_thresh = 1 # don't evaluate MRE on very small numbers -- too noisy
                if 'new_deaths' in history['nyt']:
                    mdl_hourly_new_deaths = history['nyt']['new_deaths']
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
                else:
                    mdl_prediction = mdl_IR * death_rate
                    projected_hrs = [hr + datetime.timedelta(days=death_lag) for hr in mdl_hours]
            y_true, y_pred, eval_start, eval_end = find_model_and_real_overlap_for_eval(
                real_dates, real_data, projected_hrs, mdl_prediction, compare_start_time, compare_end_time)
            if len(y_true) < 5:
                print("Fewer than 5 days of data to compare for %s; not scoring" % mode)
            else:
                score_dict['eval_start_time_%s' % mode] = eval_start
                score_dict['eval_end_time_%s' % mode] = eval_end
                score_dict['cumulative_predicted_%s' % mode] = y_pred
                score_dict['cumulative_true_%s' % mode] = y_true

                score_dict['cumulative_%s_MRE' % mode] = compute_loss(y_true, y_pred, metric='MRE', min_threshold=MRE_thresh, compare_daily_not_cumulative=False)
                score_dict['cumulative_%s_RMSE' % mode] = compute_loss(y_true, y_pred, metric='RMSE', min_threshold=None, compare_daily_not_cumulative=False)
                score_dict['cumulative_%s_MSE' % mode] = compute_loss(y_true, y_pred, metric='MSE', min_threshold=None, compare_daily_not_cumulative=False)
                # the following checks are to deal with converting a cumulative curve back to a daily
                # curve when the eval starts past the first day, which means the first entry in the
                # cumulative curve already an accumulation from multiple days, so we need to subtract
                # the cumulative value from the previous day
                if eval_start > real_dates[0]:
                    eval_start_index = real_dates.index(eval_start)
                    cumulative_day_before = real_data[eval_start_index-1]
                    y_true = y_true - cumulative_day_before
                if eval_start >= projected_hrs[24]:  # starting eval on day 2+ of simulation
                    eval_start_index = projected_hrs.index(eval_start)
                    cumulative_day_before = mdl_prediction[:, eval_start_index-24]
                    y_pred = (y_pred.T - cumulative_day_before).T
                score_dict['daily_%s_MRE' % mode] = compute_loss(y_true, y_pred, metric='MRE', min_threshold=MRE_thresh, compare_daily_not_cumulative=True)
                score_dict['daily_%s_RMSE' % mode] = compute_loss(y_true, y_pred, metric='RMSE', min_threshold=None, compare_daily_not_cumulative=True)
                score_dict['daily_%s_MSE' % mode] = compute_loss(y_true, y_pred, metric='MSE', min_threshold=None, compare_daily_not_cumulative=True)
                score_dict['daily_%s_gaussianish_negative_ll' % mode] = compute_loss(y_true, y_pred, metric='gaussianish_negative_ll', min_threshold=None, compare_daily_not_cumulative=True)

        if return_mdl_pred_and_hours and plot_mode == mode:
            return mdl_prediction, projected_hrs

        if make_plot and plot_mode == mode:
            assert(ax is not None and title is not None)
            if plot_daily_not_cumulative:
                new_projected_hrs = []
                new_mdl_prediction = []
                for hr, prediction in zip(projected_hrs, mdl_prediction.T):
                    if hr.hour == 0:
                        new_projected_hrs.append(hr)
                        new_mdl_prediction.append(prediction)
                projected_hrs = new_projected_hrs
                mdl_prediction = np.array(new_mdl_prediction).T
                mdl_prediction = get_daily_from_cumulative(mdl_prediction)
                real_data = get_daily_from_cumulative(real_data)
                if smooth_daily_cases_when_plotting:
                    real_data = apply_smoothing(real_data)

            num_seeds, _ = mdl_prediction.shape
            if num_seeds > 1:
                mean, lower_CI, upper_CI = mean_and_CIs_of_timeseries_matrix(mdl_prediction)
                model_max = max(upper_CI)
            else:
                mean = mdl_prediction[0]
                model_max = max(mean)
            real_max = max(real_data)
            daily_or_cumulative_string = 'daily' if plot_daily_not_cumulative else 'cumulative'
            if model_line_label is None:
                model_line_label = 'modeled %s %s' % (daily_or_cumulative_string, mode)
            if true_line_label is None:
                true_line_label = 'true %s %s' % (daily_or_cumulative_string, mode)
            ax.plot_date(projected_hrs, mean, linestyle='-', label=model_line_label, c=model_color, markersize=marker_size)
            if plot_real_data:
                ax.plot_date(real_dates, real_data, linestyle='-', label=true_line_label, c=real_data_color, markersize=marker_size)
            if num_seeds > 1 and plot_errorbars:
                ax.fill_between(projected_hrs, lower_CI, upper_CI, alpha=.5, color=model_color)

            interval = int(len(real_dates) / 6)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if only_plot_intersection:
                ax.set_xlim([max(min(projected_hrs), min(real_dates)), min(max(projected_hrs), max(real_dates))]) # only plot place where both lines intersect.
                right = min(max(projected_hrs), max(real_dates))
                model_max_idx = projected_hrs.index(right)
                if num_seeds > 1:
                    model_max = max(upper_CI[:model_max_idx])
                else:
                    model_max = max(mean[:model_max_idx])
                real_max_idx = real_dates.index(right)
                real_max = max(real_data[:real_max_idx])

            if plot_log:
                ax.set_yscale('log')
                ax.set_ylim([1, max(model_max, real_max)])
            else:
                ax.set_ylim([0, max(model_max, real_max)])

            if plot_legend:
                ax.legend(fontsize=legend_fontsize)

            if xticks is None:
                if x_interval is None:
                    x_interval = int(len(real_dates) / 6)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_interval))
            else:
                ax.set_xticks(xticks)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.tick_params(labelsize=tick_label_fontsize)
            if y_range is not None:
                ax.set_ylim(*y_range)
            if x_range is not None:
                ax.set_xlim(*x_range)

            if only_two_yticks:

                bot, top = ax.get_ylim()
                if plot_mode == 'cases':
                    # Round to nearest thousand
                    top = (top // 1000) * 1000
                elif plot_mode == 'deaths':
                    # Round to nearest hundred
                    top = (top // 100) * 100
                ax.set_yticks([bot, top])

            if plot_mode == 'cases':
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '{:.0f}'.format(x/1000) + 'k'))

            ax.grid(alpha=.5)
            ax.set_title(title, fontsize=title_fontsize)

    return score_dict

def compute_loss(y_true, y_pred,
                 metric='RMSE',
                 min_threshold=50,
                 compare_daily_not_cumulative=True):
    """
    This assumes that y_true and y_pred are cumulative counts.
    y_true: 1D array, the true case/death counts
    y_pred: 2D array, the predicted case/death counts over all seeds
    metric: RMSE or MRE, the loss metric
    min_threshold: the minimum number of true case/deaths that a day must have
                   to be included in eval
    compare_daily_not_cumulative: converts y_true and y_pred into daily counts
                                  and does the comparison on those instead
    """
    assert(metric in {'RMSE', 'MRE', 'MSE', 'gaussianish_negative_ll'})
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if compare_daily_not_cumulative:
        y_true = get_daily_from_cumulative(y_true)
        y_pred = get_daily_from_cumulative(y_pred)
    else:
        assert metric != 'gaussianish_negative_ll'

    if min_threshold is not None:
        idxs = y_true >= min_threshold
        if not idxs.sum() > 0:
            print(y_true)
            print("Warning: NOT ENOUGH VALUES ABOVE THRESHOLD")
            return np.nan
        y_true = y_true[idxs]
        y_pred = y_pred[:, idxs]
    if metric == 'RMSE':
        return RMSE(y_true, y_pred)
    if metric == 'MSE':
        return MSE(y_true, y_pred)
    if metric == 'gaussianish_negative_ll':
        return gaussianish_negative_ll(y_true=y_true, y_pred=y_pred)
    return MRE(y_true, y_pred)


def evaluate_all_fitted_models_for_msa(msa_name, min_timestring=None,
                                        max_timestring=None,
                                        timestrings=None,
                                       required_properties=None,
                                       required_model_kwargs=None,
                                       recompute_losses=False,
                                       key_to_sort_by='train_loss_dict_daily_cases_MRE'):

    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs
    """

    pd.set_option('max_columns', 50)
    pd.set_option('display.width', 500)

    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_properties is None:
        required_properties = {}

    if timestrings is None:
        timestrings = filter_timestrings_for_properties(
            required_properties=required_properties,
            required_model_kwargs=required_model_kwargs,
            required_data_kwargs={'MSA_name':msa_name},
            min_timestring=min_timestring,
            max_timestring=max_timestring)
        print('Found %d fitted models for %s' % (len(timestrings), msa_name))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    min_datetime = datetime.datetime(2020, 3, 1)
    max_datetime = datetime.datetime.now()  # as recent as possible
    if recompute_losses:
        nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)

    results = []
    start_time = time.time()
    for ts in timestrings:
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False,
            load_fast_results_only=(not recompute_losses))
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']
        assert data_kwargs['MSA_name'] == msa_name

        if recompute_losses:
            fast_to_load_results['loss_dict'] = compare_model_vs_real_num_cases(nyt_outcomes,
                                                   model_kwargs['min_datetime'],
                                                   model_results=model_results,
                                                   make_plot=False)

        results_for_ts = {'timestring':ts,
                         'data_kwargs':data_kwargs,
                         'model_kwargs':model_kwargs,
                         'results':model_results,
                         'experiment_to_run':experiment_to_run}

        if 'final infected fraction' in fast_to_load_results:
            results_for_ts['final infected fraction'] = fast_to_load_results['final infected fraction']

        for result_type in ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']:
            if (result_type in fast_to_load_results) and (fast_to_load_results[result_type] is not None):
                for k in fast_to_load_results[result_type]:
                    full_key = result_type + '_' + k
                    assert full_key not in results_for_ts
                    results_for_ts[full_key] = fast_to_load_results[result_type][k]

        for k in exo_kwargs:
            assert k not in results_for_ts
            results_for_ts[k] = exo_kwargs[k]
        for k in model_kwargs:
            if k == 'exogenous_model_kwargs':
                continue
            else:
                assert k not in results_for_ts
                results_for_ts[k] = model_kwargs[k]
        results.append(results_for_ts)

    end_time = time.time()
    print('Time to load and score all models: %.3fs -> %.3fs per model' %
          (end_time-start_time, (end_time-start_time)/len(timestrings)))
    results = pd.DataFrame(results)

    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results

def get_all_msas_with_fitted_models(min_timestring):
    """
    Get all the MSAs that have at least one fitted model since min_timestring.
    """
    timestrings = filter_timestrings_for_properties(min_timestring=min_timestring)
    config_dir = os.path.join(helper.FITTED_MODEL_DIR, 'data_and_model_configs')
    all_msas = set([])
    for timestring in timestrings:
        data_and_model_kwargs = pickle.load(open(os.path.join(config_dir, 'config_%s.pkl' % timestring), 'rb'))
        all_msas.add(data_and_model_kwargs['data_kwargs']['MSA_name'])
    all_msas = sorted(list(all_msas))
    print("Since %s have fitted data for %i MSAs: %s" % (min_timestring,
                                                         len(all_msas),
                                                         '\n'.join(all_msas)))
    return all_msas

def compute_overall_loss_aggregated_across_msas(min_timestring, params_to_hold_constant_across_msas, recompute_losses=False):
    """
    Finds the parameter setting with the lowest loss aggregated across MSAs [recognizing that some params can vary by MSA
    and some cannot]
    """
    # get list of MSAs
    all_msas = get_all_msas_with_fitted_models(min_timestring)

    # plot best results for each MSA.
    all_msa_results = []
    for msa in all_msas:
        msa_results = evaluate_all_fitted_models_for_msa(msa_name=msa, min_timestring=min_timestring, recompute_losses=recompute_losses)
        all_msa_results.append(msa_results)
        if recompute_losses:
            fig, ax = plt.subplots()
            print(msa)
            print(msa_results.iloc[0]['model_kwargs'])
            plot_model_fit_from_model_and_kwargs(ax,
                mdl_kwargs=msa_results.iloc[0]['model_kwargs'],
                data_kwargs=msa_results.iloc[0]['data_kwargs'],
                model_results=msa_results.iloc[0]['results'],
                plot_log=False)
            plt.show()

    # concatenate all MSA results into a single DF.
    concat_across_msas = pd.concat(all_msa_results)
    concat_across_msas['MSA_name'] = concat_across_msas['data_kwargs'].map(lambda x:x['MSA_name'])
    number_of_fitted_params_by_msa = concat_across_msas.groupby('MSA_name').size()
    print("Number of models successfully fitted by MSA:")
    print(number_of_fitted_params_by_msa)
    if len(set(number_of_fitted_params_by_msa.values)) > 1:
        print("WARNING: THE NUMBER OF SUCCESSFULLY FITTED MODELS IS NOT CONSTANT ACROSS MSAS. THIS COULD INDICATE THAT SOME MODELS FAILED TO SUCCESSFULLY FIT.")

    # for each setting of params_to_hold_constant_across_msas, compute an aggregate loss across MSAs
    # for each MSA, choose a setting of the flexible parameters which minimizes loss, and that is the loss for
    # params_to_hold_constant_across_msas
    grouped_by_param_setting = concat_across_msas.groupby(params_to_hold_constant_across_msas)
    combined_losses = []
    for param_setting, small_d in grouped_by_param_setting:
        best_loss_for_each_msa = []
        for msa in all_msas:
            best_msa_loss = sorted(small_d.loc[small_d['MSA_name'] == msa, 'loss'].values)[0]
            best_loss_for_each_msa.append(best_msa_loss)
        assert len(best_loss_for_each_msa) == len(all_msas)
        loss_for_param_setting = {'mean_loss':np.mean(best_loss_for_each_msa),
                                  'median_loss':np.median(best_loss_for_each_msa),
                                  'max_loss':np.max(best_loss_for_each_msa),
                                  'min_loss':np.min(best_loss_for_each_msa)}
        loss_for_param_setting.update(dict(zip(params_to_hold_constant_across_msas, param_setting)))
        combined_losses.append(loss_for_param_setting)
    combined_losses = (pd.DataFrame(combined_losses).sort_values(by='mean_loss')[params_to_hold_constant_across_msas +
                                                                ['mean_loss', 'median_loss',
                                                                 'max_loss', 'min_loss']])
    return combined_losses, concat_across_msas

def diagnose_why_model_files_are_too_big(original_pickle_path):
    """
    In case our model size starts exploding again, this helps debug it.
    Takes as an argument the problematic path.
    """
    d = pickle.load(open(original_pickle_path, 'rb'))
    original_size = os.path.getsize(original_pickle_path)/(1024 **2)
    print("Original size is %2.3f MB" % original_size)
    test_path = '/dfs/scratch1/safegraph_homes/all_aggregate_data/filesize_reduction_experiment.pkl'
    f = open(test_path, 'wb')
    pickle.dump(d, f)
    f.close()
    print("New file size is %2.3f MB" % (os.path.getsize(test_path)/(1024 **2)))

    for attr in d.__dict__.keys():
        object_type = str(type(getattr(d, attr)))
        if object_type in ["<class 'bool'>", "<class 'int'>", "<class 'str'>", "<class 'NoneType'>"]:
            print("Skipping %s because class %s" % (attr, object_type))
            continue
        print("Setting %s to None" % attr)
        new_d = copy.deepcopy(d)
        setattr(new_d, attr, None)
        f = open(test_path, 'wb')
        pickle.dump(new_d, f)
        f.close()
        new_size = os.path.getsize(test_path)/(1024 **2)
        print("New file size is %2.3f MB, %2.3f%% of original size" % (new_size, 100*new_size/original_size))


def print_config_as_json(data_and_model_config):
    data_and_model_config = copy.deepcopy(data_and_model_config)
    for k in data_and_model_config:
        if type(data_and_model_config[k]) is dict:
            for k1 in data_and_model_config[k]:
                data_and_model_config[k][k1] = str(data_and_model_config[k][k1])
        else:
            data_and_model_config[k] = str(data_and_model_config[k])
    print(json.dumps(data_and_model_config, indent=4, sort_keys=True))

def partition_jobs_across_computers(computer_name, configs_to_fit):
    computer_name = computer_name.replace('.stanford.edu', '')
    computers_to_use = ['trinity'] #['furiosa', 'madmax', 'madmax2', 'madmax3', 'madmax4', 'madmax5']
    computer_stats = {'rambo':288, 'trinity':144, 'furiosa':144, 'madmax':64, 'madmax2':80,  'madmax3':80, 'madmax4':80, 'madmax5':80,  'madmax6':80,  'madmax7':80}
    total_cores = sum([computer_stats[a] for a in computers_to_use])
    computer_loads = dict([(k, computer_stats[k]/total_cores) for k in computers_to_use])
    print('Partitioning up jobs among computers as follows', computer_loads)
    assert computer_name in computer_loads
    assert np.allclose(sum(computer_loads.values()), 1)
    start_idx = 0
    computers_to_configs = {}
    for computer_idx, computer in enumerate(sorted(computer_loads.keys())):
        if computer_idx == len(computer_loads) - 1:
            computers_to_configs[computer] = configs_to_fit[start_idx:]
        else:
            end_idx = start_idx + int(len(configs_to_fit) * computer_loads[computer])
            computers_to_configs[computer] = configs_to_fit[start_idx:end_idx]
            start_idx = end_idx
    assert sum([len(a) for a in computers_to_configs.values()]) == len(configs_to_fit)
    print("Assigning %i configs to %s" % (len(computers_to_configs[computer_name]), computer_name))
    return computers_to_configs[computer_name]

if __name__ == '__main__':
    # command line arguments. 
    # Basically, this script can be called two ways: either as a manager job, which generates configs and fires off a bunch of worker jobs
    # or as a worker job, which runs a single model with a single config. 
    # The command line argument manager_or_worker_job specifies which of these two usages we're using. 
    # The other important command line argument is experiment_to_run, which specifies which step of the experimental pipeline we're running. 
    # The worker jobs take additional arguments like timestring (which specifies the timestring we use to save model files) 
    # and config_idx, which specifies which config we're using. 
    parser = argparse.ArgumentParser()
    parser.add_argument('manager_or_worker_job', help='Is this the manager job or the worker job?', 
        choices=['run_many_models_in_parallel', 'fit_and_save_one_model'])
    parser.add_argument('experiment_to_run', help='The name of the experiment to run', 
        choices=['normal_grid_search', 'calibrate_r0',
                 'just_save_ipf_output', 'test_interventions',
                 'test_retrospective_counterfactuals', 'test_max_capacity_clipping',
                 'test_uniform_interpolated_reopening', 'test_uniform_proportion_of_full_reopening',
                 'rerun_best_models_and_save_cases_per_poi'])
    parser.add_argument('--timestring', type=str)
    parser.add_argument('--config_idx', type=int)
    args = parser.parse_args()

    # Less frequently used arguments. 
    config_idx_to_start_at = None
    skip_previously_fitted_kwargs = False
    min_timestring = '2020_05_23'
    min_timestring_to_load_best_fit_models_from_grid_search = '2020_06_23_15_56_15_777982' # what grid search models do we look at when loading interventions.

    config_filename = '%s_configs.pkl' % COMPUTER_WE_ARE_RUNNING_ON.replace('.stanford.edu', '')
    if args.manager_or_worker_job == 'run_many_models_in_parallel':      
        # manager job generates configs. 
        assert args.timestring is None  
        assert args.config_idx is None
        configs_to_fit = generate_data_and_model_configs(config_idx_to_start_at=config_idx_to_start_at,
            skip_previously_fitted_kwargs=skip_previously_fitted_kwargs,
            min_timestring=min_timestring,
            experiment_to_run=args.experiment_to_run,
            min_timestring_to_load_best_fit_models_from_grid_search=min_timestring_to_load_best_fit_models_from_grid_search)
        configs_to_fit = partition_jobs_across_computers(COMPUTER_WE_ARE_RUNNING_ON, configs_to_fit)
        f = open(config_filename, 'wb')
        pickle.dump(configs_to_fit, f)
        f.close()
    else:
        # worker job needs to load the list of configs and figure out which one it's running. 
        print("loading configs from %s" % config_filename)
        f = open(config_filename, 'rb')
        configs_to_fit = pickle.load(f)
        f.close()

    if args.experiment_to_run == 'normal_grid_search':
        train_test_partition = TRAIN_TEST_PARTITION
    else:
        train_test_partition = None

    if args.manager_or_worker_job == 'run_many_models_in_parallel':
        # fire off worker jobs. 
        run_many_models_in_parallel(configs_to_fit)
    elif args.manager_or_worker_job == 'fit_and_save_one_model':
        # single worker job; use command line arguments to retrieve config and timestring. 
        timestring = args.timestring
        config_idx = args.config_idx
        assert timestring is not None and config_idx is not None
        data_and_model_config = configs_to_fit[config_idx]

        print("Running single model. Kwargs are")
        print_config_as_json(data_and_model_config)

        fit_and_save_one_model(timestring,
            train_test_partition=train_test_partition,
            model_kwargs=data_and_model_config['model_kwargs'],
            data_kwargs=data_and_model_config['data_kwargs'],
            experiment_to_run=data_and_model_config['experiment_to_run'])
    else:
        raise Exception("This is not a valid way to call this method")


