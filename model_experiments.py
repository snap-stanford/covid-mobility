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
from scipy.stats import scoreatpercentile, poisson, binom
from scipy.special import logsumexp
from psutil._common import bytes2human
from scipy.stats import ttest_ind, rankdata
from scipy.sparse import hstack
import argparse
import getpass
from collections import Counter

###################################################
# Loss functions
###################################################
def MRE(y_true, y_pred):
    '''
    Computes the median relative error (MRE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the MRE is returned.
    If y_true and y_pred are 2D, e.g., predictions over multiple seeds,
    the MRE is computed per row, then averaged.
    '''
    abs_err = np.absolute(y_true - y_pred)
    rel_err = abs_err / y_true
    if len(abs_err.shape) == 1:  # this implies y_true and y_pred are 1D
        mre = np.median(rel_err)
    else:  # this implies at least one of them is 2D
        mre = np.mean(np.median(rel_err, axis=1))
    return mre

def RMSE(y_true, y_pred):
    '''
    Computes the root mean squared error (RMSE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the RMSE is returned.
    If y_true and y_pred are 2D, e.g., predictions over multiple seeds,
    the RMSE is computed per row, then averaged.
    '''
    sq_err = (y_true - y_pred) ** 2
    if len(sq_err.shape) == 1:  # this implies y_true and y_pred are 1D
        rmse = np.sqrt(np.mean(sq_err))
    else:  # this implies at least one of them is 2D
        rmse = np.sqrt(np.mean(sq_err, axis=1))
        rmse = np.mean(rmse)
    return rmse

def MSE(y_true, y_pred):
    '''
    Computes the mean squared error (MSE). y_true and y_pred should
    both be numpy arrays.
    '''
    return np.mean((y_true - y_pred) ** 2)

def gaussianish_negative_ll(y_true, y_pred, sum_or_logsumexp):
    # Gaussian but we try to estimate the variance by looking at variance over seeds
    # y_pred should be n_seeds x n_timesteps
    assert len(y_pred.shape) == 2
    assert len(y_true.shape) == 1
    sq_err = (y_true - y_pred) ** 2
    variance = np.std(y_pred, axis=0, ddof=1) ** 2 # Variance across seeds
    variance = np.clip(variance, 4, None)
    # First sum log-likelihoods over days
    ll = np.sum(-.5 * np.log(variance) - .5 * sq_err/variance, axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll

def NLL_with_var_y(y_true, y_pred, sum_or_logsumexp):
    # Gaussian with variance proportional to y_pred
    # Similar to what Li et al. (2020) tried
    # Use y_pred instead of y_true to compute the variance, since we're assuming that
    # the model predictions are the truth
    assert len(y_pred.shape) == 2
    assert len(y_true.shape) == 1
    sq_err = (y_true - y_pred) ** 2
    variance = np.clip(y_pred, 4, None)
    # First sum log-likelihoods over days
    ll = np.sum(-.5 * np.log(variance) - .5 * sq_err/variance, axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll

def NLL_with_var_ysq(y_true, y_pred, sum_or_logsumexp):
    # Gaussian with variance proportional to y_pred^2
    # Similar to what Li et al. (2020) tried
    # Use y_pred instead of y_true to compute the variance, since we're assuming that
    # the model predictions are the truth
    assert len(y_pred.shape) == 2
    assert len(y_true.shape) == 1
    sq_err = (y_true - y_pred) ** 2
    variance = np.clip((y_pred ** 2) / 4, 4, None)
    # First sum log-likelihoods over days
    ll = np.sum(-.5 * np.log(variance) - .5 * sq_err/variance, axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll

def poisson_NLL(y_true, y_pred, sum_or_logsumexp):
    # First sum log-likelihoods over days
    variance = np.clip(y_pred, 4, None)
    ll = np.sum(poisson.logpmf(y_true, variance), axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll

def binomial_NLL(y_true, y_pred, rate, sum_or_logsumexp):
    # First divide y_pred by the rate, since it's already been multiplied by this rate
    # earlier in the code
    mdl_IR = (y_pred / rate).astype(int)
    # Sum log-likelihoods over days
    ll = np.sum(binom.logpmf(y_true, mdl_IR, rate), axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll


###################################################
# Code for running one model
###################################################
def fit_disease_model_on_real_data(d,
                                   min_datetime,
                                   max_datetime,
                                   exogenous_model_kwargs,
                                   poi_attributes_to_clip,
                                   preload_poi_visits_list_filename=None,
                                   poi_cbg_visits_list=None,
                                   correct_poi_visits=True,
                                   multiply_poi_visit_counts_by_census_ratio=True,
                                   aggregate_col_to_use='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                   cbg_count_cutoff=10,
                                   cbgs_to_filter_for=None,
                                   cbg_groups_to_track=None,
                                   counties_to_track=None,
                                   include_cbg_prop_out=False,
                                   model_init_kwargs=None,
                                   simulation_kwargs=None,
                                   counterfactual_poi_opening_experiment_kwargs=None,
                                   counterfactual_retrospective_experiment_kwargs=None,
                                   return_model_without_fitting=False,
                                   return_model_and_data_without_fitting=False,
                                   model_quality_dict=None,
                                   verbose=True):
    """
    Function to prepare data as input for the disease model, and to run the disease simulation on formatted data.
    d: pandas DataFrame; POI data from SafeGraph
    min_datetime, max_datetime: DateTime objects; the first and last hour to simulate
    exogenous_model_kwargs: dict; extra arguments for Model.init_exogenous_variables()
        required keys: p_sick_at_t0, poi_psi, and home_beta
    poi_attributes_to_clip: dict; which POI attributes to clip
        required keys: clip_areas, clip_dwell_times, clip_visits
    preload_poi_visits_list_filename: str; name of file from which to load precomputed hourly networks
    poi_cbg_visits_list: list of sparse matrices; precomputed hourly networks
    correct_poi_visits: bool; whether to correct hourly visit counts with dwell time
    multiply_poi_visit_counts_by_census_ratio: bool; whether to upscale visit counts by a constant factor
        derived using Census data to try to get real visit volumes
    aggregate_col_to_use: str; the field that holds the aggregated CBG proportions for each POI
    cbg_count_cutoff: int; the minimum number of POIs a CBG must visit to be included in the model
    cbgs_to_filter_for: list; only model CBGs in this list
    cbg_groups_to_track: dict; maps group name to CBGs, will track their disease trajectories during simulation
    counties_to_track: list; names of counties, will track their disease trajectories during simulation
    include_cbg_prop_out: bool; whether to adjust the POI-CBG network based on Social Distancing Metrics (SDM);
        should only be used if precomputed poi_cbg_visits_list is not in use
    model_init_kwargs: dict; extra arguments for initializing Model
    simulation_kwargs: dict; extra arguments for Model.simulate_disease_spread()
    counterfactual_poi_opening_experiment_kwargs: dict; arguments for POI category reopening experiments
    counterfactual_retrospective_experiment_kwargs: dict; arguments for counterfactual mobility reduction experiment
    """
    assert min_datetime <= max_datetime
    assert all([k in exogenous_model_kwargs for k in ['poi_psi', 'home_beta', 'p_sick_at_t0']])
    assert all([k in poi_attributes_to_clip for k in ['clip_areas', 'clip_dwell_times', 'clip_visits']])
    assert aggregate_col_to_use in ['aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                    'aggregated_visitor_home_cbgs']
    if cbg_groups_to_track is None:
        cbg_groups_to_track = {}
    if model_init_kwargs is None:
        model_init_kwargs = {}
    if simulation_kwargs is None:
        simulation_kwargs = {}
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
    # get hour column strings
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

    # aggregate median_dwell time over weeks
    weekly_median_dwell_pattern = re.compile('2020-\d\d-\d\d.median_dwell')
    median_dwell_cols = [col for col in d.columns if re.match(weekly_median_dwell_pattern, col)]
    print('Aggregating median_dwell from %s to %s' % (median_dwell_cols[0], median_dwell_cols[-1]))
    # note: this may trigger "RuntimeWarning: All-NaN slice encountered" if a POI has all nans for median_dwell;
    # this is not a problem and will be addressed in apply_percentile_based_clipping_to_msa_df
    avg_dwell_times = d[median_dwell_cols].median(axis=1).values
    d['avg_median_dwell'] = avg_dwell_times

    # clip before dropping data so we have more POIs as basis for percentiles
    # this will also drop POIs whose sub and top categories are too small for clipping
    poi_attributes_to_clip = poi_attributes_to_clip.copy()  # copy in case we need to modify
    if poi_cbg_visits_list is not None:
        poi_attributes_to_clip['clip_visits'] = False
        print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be clipping hourly visits in dataframe')
    if poi_attributes_to_clip['clip_areas'] or poi_attributes_to_clip['clip_dwell_times'] or poi_attributes_to_clip['clip_visits']:
        d, categories_to_clip, cols_to_clip, thresholds, medians = clip_poi_attributes_in_msa_df(
            d, min_datetime, max_datetime, **poi_attributes_to_clip)
        print('After clipping, %i POIs' % len(d))

    # remove POIs with missing data
    d = d.dropna(subset=hour_cols)
    if verbose: print("After dropping for missing hourly visits, %i POIs" % len(d))
    d = d.loc[d[aggregate_col_to_use].map(lambda x:len(x.keys()) > 0)]
    if verbose: print("After dropping for missing CBG home data, %i POIs" % len(d))
    d = d.dropna(subset=['avg_median_dwell'])
    if verbose: print("After dropping for missing avg_median_dwell, %i POIs" % len(d))

    # reindex CBGs
    poi_cbg_proportions = d[aggregate_col_to_use].values  # an array of dicts; each dict represents CBG distribution for POI
    all_cbgs = [a for b in poi_cbg_proportions for a in b.keys()]
    cbg_counts = Counter(all_cbgs).most_common()
    # only keep CBGs that have visited at least this many POIs
    all_unique_cbgs = [cbg for cbg, count in cbg_counts if count >= cbg_count_cutoff]
    if cbgs_to_filter_for is not None:
        print("Prior to filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
        all_unique_cbgs = [a for a in all_unique_cbgs if a in cbgs_to_filter_for]
        print("After filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))

    # order CBGs lexicographically
    all_unique_cbgs = sorted(all_unique_cbgs)
    N = len(all_unique_cbgs)
    if verbose: print("After dropping CBGs that appear in < %i POIs, %i CBGs (%2.1f%%)" %
          (cbg_count_cutoff, N, 100.*N/len(cbg_counts)))
    cbgs_to_idxs = dict(zip(all_unique_cbgs, range(N)))

    # convert data structures with CBG names to CBG indices
    poi_cbg_proportions_int_keys = []
    kept_poi_idxs = []
    E = 0   # number of connected POI-CBG pairs
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
    if poi_cbg_visits_list is not None:
        expected_M, expected_N = poi_cbg_visits_list[0].shape
        assert M == expected_M
        assert N == expected_N

    cbg_idx_groups_to_track = {}
    for group in cbg_groups_to_track:
        cbg_idx_groups_to_track[group] = [
            cbgs_to_idxs[a] for a in cbg_groups_to_track[group] if a in cbgs_to_idxs]
        if verbose: print(f'{len(cbg_groups_to_track[group])} CBGs in {group} -> matched {len(cbg_idx_groups_to_track[group])} ({(len(cbg_idx_groups_to_track[group]) / len(cbg_groups_to_track[group])):.3f})')

    # get POI-related variables
    d = d.iloc[kept_poi_idxs]
    poi_subcategory_types = d['sub_category'].values
    poi_areas = d['safegraph_computed_area_in_square_feet'].values
    poi_dwell_times = d['avg_median_dwell'].values
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    print('Dwell time correction factors: mean = %.2f, min = %.2f, max = %.2f' %
          (np.mean(poi_dwell_time_correction_factors), min(poi_dwell_time_correction_factors), max(poi_dwell_time_correction_factors)))
    poi_time_counts = d[hour_cols].values
    if correct_poi_visits:
        if poi_cbg_visits_list is not None:
            print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be applying correction to hourly visits in dataframe')
        else:
            print('Correcting POI hourly visit vectors...')
            new_poi_time_counts = []
            for i, (visit_vector, dwell_time) in enumerate(list(zip(poi_time_counts, poi_dwell_times))):
                new_poi_time_counts.append(correct_visit_vector(visit_vector, dwell_time))
            poi_time_counts = np.array(new_poi_time_counts)
            d[hour_cols] = poi_time_counts
            new_hourly_visit_count = np.sum(poi_time_counts)
            print('After correcting, %.2f hourly visits' % new_hourly_visit_count)

    # get CBG-related variables from census data
    print('2. Processing ACS data...')
    acs_d = helper.load_and_reconcile_multiple_acs_data()
    cbgs_to_census_pops = dict(zip(acs_d['census_block_group'].values,
                                   acs_d['total_cbg_population_2018_1YR'].values))  # use most recent population data
    cbg_sizes = np.array([cbgs_to_census_pops[a] for a in all_unique_cbgs])
    assert np.sum(np.isnan(cbg_sizes)) == 0
    if verbose:
        print('CBGs: median population size = %d, sum of population sizes = %d' %
          (np.median(cbg_sizes), np.sum(cbg_sizes)))

    if multiply_poi_visit_counts_by_census_ratio:
        # Get overall undersampling factor.
        # Basically we take ratio of ACS US population to SafeGraph population in Feb 2020.
        # SafeGraph thinks this is reasonable.
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

    if counties_to_track is not None:
        print('Found %d counties to track...' % len(counties_to_track))
        county2cbgs = {}
        for county in counties_to_track:
            county_cbgs = acs_d[acs_d['county_code'] == county]['census_block_group'].values
            orig_len = len(county_cbgs)
            county_cbgs = sorted(set(county_cbgs).intersection(set(all_unique_cbgs)))
            if orig_len > 0:
                coverage = len(county_cbgs) / orig_len
                if coverage < 0.8:
                    print('Low coverage warning: only modeling %d/%d (%.1f%%) of the CBGs in %s' %
                          (len(county_cbgs), orig_len, 100. * coverage, county))
            if len(county_cbgs) > 0:
                county_cbg_idx = np.array([cbgs_to_idxs[a] for a in county_cbgs])
                county2cbgs[county] = (county_cbgs, county_cbg_idx)
        print('Modeling CBGs from %d of the counties' % len(county2cbgs))
    else:
        county2cbgs = None

    # turn off warnings temporarily so that using > or <= on np.nan does not cause warnings
    np.warnings.filterwarnings('ignore')
    cbg_idx_to_track = set(range(N))  # include all CBGs
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

        if county2cbgs is not None:
            above_median_in_county = []
            below_median_in_county = []
            for county in county2cbgs:
                county_cbgs, cbg_idx = county2cbgs[county]
                attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in county_cbgs])
                non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
                median_cutoff = np.median(non_nan_vals)
                above_median_idx = cbg_idx[np.where(attribute_vals > median_cutoff)[0]]
                above_median_idx = list(set(above_median_idx).intersection(cbg_idx_to_track))
                above_median_in_county.extend(above_median_idx)
                below_median_idx = cbg_idx[np.where(attribute_vals <= median_cutoff)[0]]
                below_median_idx = list(set(below_median_idx).intersection(cbg_idx_to_track))
                below_median_in_county.extend(below_median_idx)
            cbg_idx_groups_to_track[f'{attribute}_above_median_in_own_county'] = above_median_in_county
            cbg_idx_groups_to_track[f'{attribute}_below_median_in_own_county'] = below_median_in_county
    np.warnings.resetwarnings()

    if include_cbg_prop_out:
        model_days = helper.list_datetimes_in_range(min_datetime, max_datetime)
        cols_to_keep = ['%s.%s.%s' % (dt.year, dt.month, dt.day) for dt in model_days]
        print('Giving model prop out for %s to %s' % (cols_to_keep[0], cols_to_keep[-1]))
        assert((len(cols_to_keep) * 24) == len(hour_cols))
        print('Loading Social Distancing Metrics and computing CBG prop out per day: warning, this could take a while...')
        sdm_mdl = helper.load_social_distancing_metrics(model_days)
        cbg_day_prop_out = helper.compute_cbg_day_prop_out(sdm_mdl, all_unique_cbgs)
        assert(len(cbg_day_prop_out) == len(all_unique_cbgs))
        # sort lexicographically, like all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out.sort_values(by='census_block_group')
        assert list(cbg_day_prop_out['census_block_group'].values) == all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out[cols_to_keep].values
    else:
        cbg_day_prop_out = None

    # If trying to get the counterfactual where social activity doesn't change, just repeat first week of dataset.
    # We put this in exogenous_model_kwargs because it actually affects how the model runs, not just the data input.
    if 'just_compute_r0' in exogenous_model_kwargs and exogenous_model_kwargs['just_compute_r0']:
        print('Running model to compute r0 -> looping first week visit counts')
        # simulate out 15 weeks just so we are sure all cases are gone.
        max_datetime = min_datetime + datetime.timedelta(hours=(168*15)-1)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        if poi_cbg_visits_list is not None:
            assert len(poi_cbg_visits_list) >= 168  # ensure that we have at least a week to model
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

    # If we want to run counterfactual reopening simulations
    intervention_cost = None
    if counterfactual_poi_opening_experiment_kwargs is not None:
        if poi_cbg_visits_list is None:
            raise Exception('Missing poi_cbg_visits_list; reopening experiments should be run with IPF output')
        extra_weeks_to_simulate = counterfactual_poi_opening_experiment_kwargs['extra_weeks_to_simulate']
        assert extra_weeks_to_simulate >= 0
        orig_num_hours = len(all_hours)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime + datetime.timedelta(hours=168 * extra_weeks_to_simulate))
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        intervention_datetime = counterfactual_poi_opening_experiment_kwargs['intervention_datetime']
        assert(intervention_datetime in all_hours)
        intervention_hour_idx = all_hours.index(intervention_datetime)
        if 'top_category' in counterfactual_poi_opening_experiment_kwargs:
            top_category = counterfactual_poi_opening_experiment_kwargs['top_category']
        else:
            top_category = None
        if 'sub_category' in counterfactual_poi_opening_experiment_kwargs:
            sub_category = counterfactual_poi_opening_experiment_kwargs['sub_category']
        else:
            sub_category = None
        poi_categories = d[['top_category', 'sub_category']]

        # must have one but not both of these arguments
        assert (('alpha' in counterfactual_poi_opening_experiment_kwargs) + ('full_activity_alpha' in counterfactual_poi_opening_experiment_kwargs)) == 1
        # the original alpha - post-intervention is interpolation between no reopening and full activity
        if 'alpha' in counterfactual_poi_opening_experiment_kwargs:
            alpha = counterfactual_poi_opening_experiment_kwargs['alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list,
                                        poi_categories, poi_areas, all_hours, intervention_hour_idx,
                                        alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=True)
        # post-intervention is alpha-percent of full activity (no interpolation)
        else:
            alpha = counterfactual_poi_opening_experiment_kwargs['full_activity_alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list,
                                        poi_categories, poi_areas, all_hours, intervention_hour_idx,
                                        alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=False)

        # should be used in tandem with alpha or full_activity_alpha, since the timeseries is extended
        # in those blocks; this part just caps post-intervention visits to alpha-percent of max capacity
        if 'max_capacity_alpha' in counterfactual_poi_opening_experiment_kwargs:
            max_capacity_alpha = counterfactual_poi_opening_experiment_kwargs['max_capacity_alpha']
            assert max_capacity_alpha >= 0 and max_capacity_alpha <= 1
            poi_visits = np.zeros((M, orig_num_hours))   # num pois x num hours
            for t, poi_cbg_visits in enumerate(poi_cbg_visits_list[:orig_num_hours]):
                poi_visits[:, t] = poi_cbg_visits @ np.ones(N)
            max_per_poi = np.max(poi_visits, axis=1)  # get historical max capacity per POI
            alpha_max_per_poi = np.clip(max_capacity_alpha * max_per_poi, 1e-10, None)  # so that we don't divide by 0
            orig_total_activity = 0
            capped_total_activity = 0
            for t in range(intervention_hour_idx, len(poi_cbg_visits_list)):
                poi_cbg_visits = poi_cbg_visits_list[t]
                num_visits_per_poi = poi_cbg_visits @ np.ones(N)
                orig_total_activity += np.sum(num_visits_per_poi)
                ratio_per_poi = num_visits_per_poi / alpha_max_per_poi
                clipping_idx = ratio_per_poi > 1  # identify which POIs need to be clipped
                poi_multipliers = np.ones(M)
                poi_multipliers[clipping_idx] = 1 / ratio_per_poi[clipping_idx]
                adjusted_poi_cbg_visits = poi_cbg_visits.transpose().multiply(poi_multipliers).transpose().tocsr()
                capped_total_activity += np.sum(adjusted_poi_cbg_visits @ np.ones(N))
                poi_cbg_visits_list[t] = adjusted_poi_cbg_visits
            print('Finished capping visits at %.1f%% of max capacity -> kept %.4f%% of visits' %
                  (100. * max_capacity_alpha, 100 * capped_total_activity / orig_total_activity))
            intervention_cost['total_activity_after_max_capacity_capping'] = capped_total_activity

    if counterfactual_retrospective_experiment_kwargs is not None:
        # must have one but not both of these arguments
        assert (('distancing_degree' in counterfactual_retrospective_experiment_kwargs) + ('shift_in_days' in counterfactual_retrospective_experiment_kwargs)) == 1
        if poi_cbg_visits_list is None:
            raise Exception('Retrospective experiments are only implemented for when poi_cbg_visits_list is precomputed')
        if 'distancing_degree' in counterfactual_retrospective_experiment_kwargs:
            distancing_degree = counterfactual_retrospective_experiment_kwargs['distancing_degree']
            poi_cbg_visits_list = apply_distancing_degree(poi_cbg_visits_list, distancing_degree)
            print('Modified poi_cbg_visits_list for retrospective experiment: distancing_degree = %s.' % distancing_degree)
        else:
            shift_in_days = counterfactual_retrospective_experiment_kwargs['shift_in_days']
            poi_cbg_visits_list = apply_shift_in_days(poi_cbg_visits_list, shift_in_days)
            print('Modified poi_cbg_visits_list for retrospective experiment: shifted by %d days.' % shift_in_days)

    print('Total time to prep data: %.3fs' % (time.time() - t0))

    # feed everything into model.
    m = Model(**model_init_kwargs)
    m.init_exogenous_variables(poi_cbg_proportions=poi_cbg_proportions_int_keys,
                               poi_time_counts=poi_time_counts,
                               poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               all_unique_cbgs=all_unique_cbgs,
                               cbgs_to_idxs=cbgs_to_idxs,
                               all_states=all_states,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
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

def clip_poi_attributes_in_msa_df(d, min_datetime, max_datetime,
                                  clip_areas, clip_dwell_times, clip_visits,
                                  area_below=AREA_CLIPPING_BELOW,
                                  area_above=AREA_CLIPPING_ABOVE,
                                  dwell_time_above=DWELL_TIME_CLIPPING_ABOVE,
                                  visits_above=HOURLY_VISITS_CLIPPING_ABOVE,
                                  subcat_cutoff=SUBCATEGORY_CLIPPING_THRESH,
                                  topcat_cutoff=TOPCATEGORY_CLIPPING_THRESH):
    '''
    Deal with POI outliers by clipping their hourly visits, dwell times, and physical areas
    to some percentile of the corresponding distribution for each POI category.
    '''
    attr_cols = []
    if clip_areas:
        attr_cols.append('safegraph_computed_area_in_square_feet')
    if clip_dwell_times:
        attr_cols.append('avg_median_dwell')
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    if clip_visits:
        attr_cols.extend(hour_cols)
    assert all([col in d.columns for col in attr_cols])
    print('Clipping areas: %s (below=%d, above=%d), clipping dwell times: %s (above=%d), clipping visits: %s (above=%d)' %
          (clip_areas, area_below, area_above, clip_dwell_times, dwell_time_above, clip_visits, visits_above))

    subcats = []
    left_out_subcats = []
    indices_covered = []
    subcategory2idx = d.groupby('sub_category').indices
    for cat, idx in subcategory2idx.items():
        if len(idx) >= subcat_cutoff:
            subcats.append(cat)
            indices_covered.extend(idx)
        else:
            left_out_subcats.append(cat)
    num_subcat_pois = len(indices_covered)

    # group by top_category for POIs whose sub_category's are too small
    topcats = []
    topcategory2idx = d.groupby('top_category').indices
    remaining_pois = d[d['sub_category'].isin(left_out_subcats)]
    necessary_topcats = set(remaining_pois.top_category.unique())  # only necessary to process top_category's that have at least one remaining POI
    for cat, idx in topcategory2idx.items():
        if cat in necessary_topcats and len(idx) >= topcat_cutoff:
            topcats.append(cat)
            new_idx = np.array(list(set(idx) - set(indices_covered)))  # POIs that are not covered by sub_category clipping
            assert len(new_idx) > 0
            topcategory2idx[cat] = (idx, new_idx)
            indices_covered.extend(new_idx)

    print('Found %d sub-categories with >= %d POIs and %d top categories with >= %d POIs -> covers %d POIs' %
          (len(subcats), subcat_cutoff, len(topcats), topcat_cutoff, len(indices_covered)))
    kept_visits = np.nansum(d.iloc[indices_covered][hour_cols].values)
    all_visits = np.nansum(d[hour_cols].values)
    lost_visits = all_visits - kept_visits
    lost_pois = len(d) - len(indices_covered)
    print('Could not cover %d/%d POIs (%.1f%% POIs, %.1f%% hourly visits) -> dropping these POIs' %
          (lost_pois, len(d), 100. * lost_pois/len(d), 100 * lost_visits / all_visits))
    if lost_pois / len(d) > .03:
        raise Exception('Dropping too many POIs during clipping phase')

    all_cats = topcats + subcats  # process top categories first so sub categories will compute percentiles on raw data
    new_data = np.array(d[attr_cols].copy().values)  # n_pois x n_cols_to_clip
    thresholds = np.zeros((len(all_cats), len(attr_cols)+1))  # clipping thresholds for category x attribute
    medians = np.zeros((len(all_cats), len(attr_cols)))  # medians for category x attribute
    indices_processed = []
    for i, cat in enumerate(all_cats):
        if i < len(topcats):
            cat_idx, new_idx = topcategory2idx[cat]
        else:
            cat_idx = subcategory2idx[cat]
            new_idx = cat_idx
        indices_processed.extend(new_idx)
        first_col_idx = 0  # index of first column for this attribute

        if clip_areas:
            cat_areas = new_data[cat_idx, first_col_idx]  # compute percentiles on entire category
            min_area = np.nanpercentile(cat_areas, area_below)
            max_area = np.nanpercentile(cat_areas, area_above)
            median_area = np.nanmedian(cat_areas)
            thresholds[i][first_col_idx] = min_area
            thresholds[i][first_col_idx+1] = max_area
            medians[i][first_col_idx] = median_area
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], min_area, max_area)
            first_col_idx += 1

        if clip_dwell_times:
            cat_dwell_times = new_data[cat_idx, first_col_idx]
            max_dwell_time = np.nanpercentile(cat_dwell_times, dwell_time_above)
            median_dwell_time = np.nanmedian(cat_dwell_times)
            thresholds[i][first_col_idx+1] = max_dwell_time
            medians[i][first_col_idx] = median_dwell_time
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], None, max_dwell_time)
            first_col_idx += 1

        if clip_visits:
            col_idx = np.arange(first_col_idx, first_col_idx+len(hour_cols))
            assert col_idx[-1] == (len(attr_cols)-1)
            orig_visits = new_data[cat_idx][:, col_idx].copy()  # need to copy bc will modify
            orig_visits[orig_visits == 0] = np.nan  # want percentile over positive visits
            # can't take percentile of col if it is all 0's or all nan's
            cols_to_process = col_idx[np.sum(~np.isnan(orig_visits), axis=0) > 0]
            max_visits_per_hour = np.nanpercentile(orig_visits[:, cols_to_process-first_col_idx], visits_above, axis=0)
            assert np.sum(np.isnan(max_visits_per_hour)) == 0
            thresholds[i][cols_to_process + 1] = max_visits_per_hour
            medians[i][cols_to_process] = np.nanmedian(orig_visits[:, cols_to_process-first_col_idx], axis=0)

            orig_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            orig_attributes = new_data[new_idx]  # return to un-modified version
            orig_attributes[:, cols_to_process] = np.clip(orig_attributes[:, cols_to_process], None, max_visits_per_hour)
            new_data[new_idx] = orig_attributes
            new_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            print('%s -> has %d POIs, processed %d POIs, %d visits before clipping, %d visits after clipping' %
              (cat, len(cat_idx), len(new_idx), orig_visit_sum, new_visit_sum))
        else:
            print('%s -> has %d POIs, processed %d POIs' % (cat, len(cat_idx), len(new_idx)))

    assert len(indices_processed) == len(set(indices_processed))  # double check that we only processed each POI once
    assert set(indices_processed) == set(indices_covered)  # double check that we processed the POIs we expected to process
    new_d = d.iloc[indices_covered].copy()
    new_d[attr_cols] = new_data[indices_covered]
    return new_d, all_cats, attr_cols, thresholds, medians

def apply_interventions_to_poi_cbg_matrices(poi_cbg_visits_list, poi_categories, poi_areas,
                                            new_all_hours, intervention_hour_idx,
                                            alpha, extra_weeks_to_simulate,
                                            top_category=None, sub_category=None,
                                            interpolate=True):
    '''
    Simulates hypothetical mobility patterns by editing visit matrices.
    '''
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

def apply_distancing_degree(poi_cbg_visits_list, distancing_degree):
    """
    After the first week of March, assume that activity is an interpolation between true activity and first-week-of-March activity
    """
    new_visits_list = []
    for i, m in enumerate(poi_cbg_visits_list):
        if i < 168:  # first week
            new_visits_list.append(m.copy())
        else:
            first_week_m = poi_cbg_visits_list[i % 168]
            mixture = first_week_m.multiply(1-distancing_degree) + m.multiply(distancing_degree)
            new_visits_list.append(mixture.copy())
    return new_visits_list

def apply_shift_in_days(poi_cbg_visits_list, shift_in_days):
    """
    Shift entire visits timeline shift_in_days days forward or backward,
    filling in the beginning or end as necessary with data from the first or last week.
    """
    new_visits_list = []
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
                # fill in with the last part of the first week.
                # so eg if shift_in_hours is 72, we take the last 72 hours of the first week.
                first_week_idx = (168 - shift_in_hours + i) % 168

                # alternate, more complex computation as sanity check.
                distance_from_start = (shift_in_hours - i) % 168
                first_week_idx_2 = (168 - distance_from_start) % 168

                assert first_week_idx_2 == first_week_idx
                new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
            else:
                new_visits_list.append(poi_cbg_visits_list[i-shift_in_hours].copy())
    assert len(new_visits_list) == len(poi_cbg_visits_list)
    return new_visits_list

def get_ipf_filename(msa_name, min_datetime, max_datetime, clip_visits, correct_visits=True):
    """
    Get the filename matching these parameters of IPF.
    """
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

def fit_and_save_one_model(timestring,
                           model_kwargs,
                           data_kwargs,
                           d=None,
                           experiment_to_run=None,
                           train_test_partition=None,
                           filter_for_cbgs_in_msa=False):
    '''
    Fits one model, saves its results and evaluations of the results.
    timestring: str; to use in filenames to identify the model and its config;
        if None, then the model is not saved
    model_kwargs: dict; arguments to use for fit_disease_model_on_real_data
        required keys: min_datetime, max_datetime, exogenous_model_kwargs, poi_attributes_to_clip
    data_kwargs: dict; arguments for the data; required to have key 'MSA_name'
    d: pandas DataFrame; the dataframe for the MSA pois; if None, then the dataframe is loaded
        within the function
    experiment_to_run: str; name of experiment to run
    train_test_partition: DateTime object; the first hour of test; if included, then losses are saved
        separately for train and test dates
    filter_for_cbgs_in_msa: bool; whether to only model CBGs in the MSA
    '''
    assert all([k in model_kwargs for k in ['min_datetime', 'max_datetime', 'exogenous_model_kwargs',
                                            'poi_attributes_to_clip']])
    assert 'MSA_name' in data_kwargs
    t0 = time.time()
    return_without_saving = False
    if timestring is None:
        print("Fitting single model. Timestring is none so not saving model and just returning fitted model.")
        return_without_saving = True
    else:
        print("Fitting single model. Results will be saved using timestring %s" % timestring)
    if d is None:  # load data
        d = helper.load_dataframe_for_individual_msa(**data_kwargs)
    nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(data_kwargs['MSA_name'])
    if 'counties_to_track' not in model_kwargs:
        model_kwargs['counties_to_track'] = msa_counties
    cbg_groups_to_track = {}
    cbg_groups_to_track['nyt'] = nyt_cbgs
    if filter_for_cbgs_in_msa:
        print("Filtering for %i CBGs within MSA %s" % (len(msa_cbgs), data_kwargs['MSA_name']))
        cbgs_to_filter_for = set(msa_cbgs) # filter for CBGs within MSA
    else:
        cbgs_to_filter_for = None

    correct_visits = model_kwargs['correct_visits'] if 'correct_visits' in model_kwargs else True  # default to True
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
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'],
            correct_visits=correct_visits)
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
            correct_visits=correct_visits)
        print('Saving IPF output in', ipf_filename)
        ipf_file = open(ipf_filename, 'wb')
        pickle.dump(fitted_model.poi_cbg_visit_history, ipf_file)
        ipf_file.close()
        print('Time to save pickle = %.2fs' % (time.time() - pickle_start_time))
        print('Size of pickle: %.2f MB' % (os.path.getsize(ipf_filename) / (1024**2)))
        return

    if return_without_saving:
        return fitted_model

    # evaluate results broken down by race and SES.
    plot_path = os.path.join(helper.FITTED_MODEL_DIR, 'ses_race_plots', 'ses_race_plot_%s.pdf' % timestring)
    ses_race_results = make_slir_race_ses_plot(fitted_model, path_to_save=plot_path)
    fitted_model.SES_RACE_RESULTS = ses_race_results

    # Save model
    mdl_path = os.path.join(FITTED_MODEL_DIR, 'full_models', 'fitted_model_%s.pkl' % timestring)
    print("Saving model at %s..." % mdl_path)
    file = open(mdl_path, 'wb')
    fitted_model.save(file)
    file.close()

    model_results_to_save_separately = {}
    for attr_to_save_separately in ['history', 'CBGS_TO_IDXS']:
        model_results_to_save_separately[attr_to_save_separately] = getattr(fitted_model, attr_to_save_separately)
    model_results_to_save_separately['ses_race_results'] = ses_race_results

    if SAVE_MODEL_RESULTS_SEPARATELY:
        # Save some smaller model results for quick(er) loading. For really fast stuff, like losses (numerical results only) we store separately.
        print("Saving model results...")
        file = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'wb')
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

    file = open(os.path.join(FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()

    # Save kwargs.
    data_and_model_kwargs = {'model_kwargs':model_kwargs, 'data_kwargs':data_kwargs, 'experiment_to_run':experiment_to_run}
    file = open(os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'wb')
    pickle.dump(data_and_model_kwargs, file)
    file.close()
    print("Successfully fitted and saved model and data_and_model_kwargs; total time taken %2.3f seconds" % (time.time() - t0))
    return fitted_model

def load_model_and_data_from_timestring(timestring, verbose=False, load_original_data=False,
                                        load_full_model=False, load_fast_results_only=True,
                                        load_filtered_data_model_was_fitted_on=False,
                                        old_directory=False):

    if verbose:
        print("Loading model from timestring %s" % timestring)
    if old_directory:
        model_dir = OLD_FITTED_MODEL_DIR
    else:
        model_dir = FITTED_MODEL_DIR
    f = open(os.path.join(model_dir, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')
    data_and_model_kwargs = pickle.load(f)
    f.close()
    model = None
    model_results = None
    f = open(os.path.join(model_dir, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'rb')
    fast_to_load_results = pickle.load(f)
    f.close()

    if not load_fast_results_only:
        if SAVE_MODEL_RESULTS_SEPARATELY:
            f = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'rb')
            model_results = pickle.load(f)
            f.close()

        if load_full_model:
            f = open(os.path.join(model_dir, 'full_models', 'fitted_model_%s.pkl' % timestring), 'rb')
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

###################################################
# Code for running many models in parallel
###################################################
def generate_data_and_model_configs(msas_to_fit=10,
                                    config_idx_to_start_at=None,
                                    skip_previously_fitted_kwargs=False,
                                    min_timestring=None,
                                    min_timestring_to_load_best_fit_models_from_grid_search=None,
                                    experiment_to_run='normal_grid_search',
                                    how_to_select_best_grid_search_models=None,
                                    max_models_to_take_per_msa=MAX_MODELS_TO_TAKE_PER_MSA,
                                    acceptable_loss_tolerance=ACCEPTABLE_LOSS_TOLERANCE):
    """
    Generates the set of parameter configurations for a given experiment.
    MSAs to fit: how many MSAs we will focus on.
    config_idx_to_start_at: how many configs we should skip.
    """
    # this controls what parameters we search over.
    config_generation_start_time = time.time()

    if skip_previously_fitted_kwargs:
        assert min_timestring is not None
        previously_fitted_timestrings = filter_timestrings_for_properties(min_timestring=min_timestring)
        previously_fitted_data_and_model_kwargs = [pickle.load(open(os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')) for timestring in previously_fitted_timestrings]
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
    min_datetime = datetime.datetime(2020, 3, 1, 0)  # min_datetime is fixed
    date_cols = [helper.load_date_col_as_date(a) for a in d.columns]
    date_cols = [a for a in date_cols if a is not None]
    max_date = max(date_cols)
    max_datetime = datetime.datetime(max_date.year, max_date.month, max_date.day, 23)  # latest hour
    print('Min datetime: %s. Max datetime: %s.' % (min_datetime, max_datetime))

    # Generate model kwargs. How exactly we do this depends on which experiments we're running.
    num_seeds = 30
    configs_with_changing_params = []
    if experiment_to_run == 'just_save_ipf_output':
        model_kwargs = [{'min_datetime':min_datetime,
                         'max_datetime':max_datetime,
                         'exogenous_model_kwargs': {  # could be anything, will not affect IPF
                            'home_beta':1e-2,
                            'poi_psi':1000,
                            'p_sick_at_t0':1e-4,
                            'just_compute_r0':False,
                          },
                          'poi_attributes_to_clip':{
                              'clip_areas':True,
                              'clip_dwell_times':True,
                              'clip_visits':True
                          },
                          'model_init_kwargs':{
                            'ipf_final_match':'poi',
                            'ipf_num_iter':100,
                            'num_seeds':num_seeds,
                          },
                          'include_cbg_prop_out':True}]

    elif experiment_to_run == 'normal_grid_search':
        p_sicks = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
        home_betas = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
        #[0.001, 0.002, 0.005, 0.008, 0.01, 0.012]
        poi_psis = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)
        #[a for a in np.arange(500, 5001, 500)]
        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    configs_with_changing_params.append({'home_beta':home_beta, 'poi_psi':poi_psi, 'p_sick_at_t0':p_sick})

        # ablation analyses.
        for home_beta in np.linspace(0.005, 0.04, 20):#sorted(list(set(home_betas + list(np.arange(0.0001, 0.002, 0.0001)) + [0.0015, 0.003, 0.004]))):
            for p_sick in p_sicks:
                configs_with_changing_params.append({'home_beta':home_beta, 'poi_psi':0, 'p_sick_at_t0':p_sick})

    elif experiment_to_run == 'grid_search_aggregate_mobility':
        p_sicks = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
        home_betas = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
                                 BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
        poi_psis = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'],
                               BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)
        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    configs_with_changing_params.append({'home_beta':home_beta, 'poi_psi':poi_psi, 'p_sick_at_t0':p_sick})

    elif experiment_to_run == 'calibrate_r0':
        home_betas = [5e-2, 0.02, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 0]
        poi_psis = [20000, 15000, 10000,7500,6000, 5000,4500,4000,3500,3000,2500,2000,1500,1000,500,250,100]
        for home_beta in home_betas:
            configs_with_changing_params.append({'home_beta':home_beta, 'poi_psi':2500, 'p_sick_at_t0':1e-4})
        for poi_psi in poi_psis:
            configs_with_changing_params.append({'home_beta':0.001, 'poi_psi':poi_psi, 'p_sick_at_t0':1e-4})

    elif experiment_to_run == 'calibrate_r0_aggregate_mobility':
        # home beta range will be the same as normal experiment
        poi_psis = [50, 25, 10, 5,  1, 0.5, 0.1, 0.005, 0.001]
        for poi_psi in poi_psis:
            configs_with_changing_params.append({'home_beta':0.001, 'poi_psi':poi_psi, 'p_sick_at_t0':1e-4})

    # experiments that require the best fit models
    best_models_experiments = {
        'test_interventions',
        'test_retrospective_counterfactuals',
        'test_max_capacity_clipping',
        'test_uniform_proportion_of_full_reopening',
        'rerun_best_models_and_save_cases_per_poi'}
    if experiment_to_run in best_models_experiments:
        # here model and data kwargs are entwined, so we can't just take the outer product of model_kwargs and data_kwargs.
        # this is because we load the best fitting model for each MSA.
        list_of_data_and_model_kwargs = []
        poi_categories_to_examine = 20
        if how_to_select_best_grid_search_models == 'daily_cases_rmse':
            key_to_sort_by = 'loss_dict_daily_cases_RMSE'
        elif how_to_select_best_grid_search_models == 'daily_deaths_rmse':
            key_to_sort_by = 'loss_dict_daily_deaths_RMSE'
        elif how_to_select_best_grid_search_models == 'daily_cases_poisson':
            key_to_sort_by = 'loss_dict_daily_cases_poisson_NLL_thres-10_sum'
        else:
            raise Exception("Not a valid means of selecting best-fit models")
        print("selecting best grid search models using criterion %s" % how_to_select_best_grid_search_models)

        # examine most common POI categories.
        most_visited_poi_subcategories = get_list_of_poi_subcategories_with_most_visits(n_poi_categories=poi_categories_to_examine)

        # Get list of all fitted models.
        model_timestrings, model_msas = filter_timestrings_for_properties(
            min_timestring=min_timestring_to_load_best_fit_models_from_grid_search,
            required_properties={'experiment_to_run':'normal_grid_search'},
            return_msa_names=True)
        print("Found %i models after %s" % (len(model_timestrings), min_timestring_to_load_best_fit_models_from_grid_search))
        timestring_msa_df = pd.DataFrame({'model_timestring':model_timestrings, 'model_msa':model_msas})
        n_models_for_msa_prior_to_quality_filter = None
        for row in data_kwargs:
            msa_t0 = time.time()
            msa_name = row['MSA_name']
            timestrings_for_msa = timestrings=list(
                timestring_msa_df.loc[timestring_msa_df['model_msa'] == msa_name, 'model_timestring'].values)
            print("Evaluating %i timestrings for %s" % (len(timestrings_for_msa), msa_name))
            best_msa_models = evaluate_all_fitted_models_for_msa(msa_name, timestrings=timestrings_for_msa)

            best_msa_models = best_msa_models.loc[(best_msa_models['experiment_to_run'] == 'normal_grid_search') &
            (best_msa_models['poi_psi'] > 0)].sort_values(by=key_to_sort_by)

            if n_models_for_msa_prior_to_quality_filter is None:
                n_models_for_msa_prior_to_quality_filter = len(best_msa_models) # make sure nothing weird happening / no duplicate models.
            else:
                assert len(best_msa_models) == n_models_for_msa_prior_to_quality_filter

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
                                      'how_to_select_best_grid_search_models':how_to_select_best_grid_search_models,
                                      'ratio_of_%s_to_that_of_best_fitting_model' % key_to_sort_by:loss_ratio,
                                      'model_timestring':best_msa_models.iloc[i]['timestring']}
                _, kwargs_i, _, _, _ = load_model_and_data_from_timestring(best_msa_models.iloc[i]['timestring'], load_fast_results_only=True)
                kwargs_i['experiment_to_run'] = experiment_to_run
                del kwargs_i['model_kwargs']['counties_to_track']

                if experiment_to_run == 'test_retrospective_counterfactuals':
                    # LOOKING AT THE PAST.
                    # what if we had only done x% of social distancing?
                    # degree represents what percentage of social distancing to keep - we don't need to test 1
                    # because that is what actually happened
                    for degree in [0, 0.25, 0.5, 0.75]:
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'distancing_degree':degree}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                    # what if we shifted the timeseries by x days?
                    for shift in [-7, -3, 3, 7]:  # how many days to shift
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'shift_in_days':shift}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
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
                            kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()

                            kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                            list_of_data_and_model_kwargs.append(kwarg_copy)

                elif experiment_to_run == 'test_max_capacity_clipping':
                    # FUTURE EXPERIMENTS: reopen fully but clip at alpha-proportion of max capacity
                    for alpha in np.arange(.1, 1.1, .1):
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                                               'extra_weeks_to_simulate':4,
                                               'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                               'alpha':1,  # assume full activity before clipping
                                               'max_capacity_alpha':alpha}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
                    # FUTURE EXPERIMENTS: test uniform reopening on all pois, simple proportion of pre-lockdown activity
#                     for alpha in np.arange(.3, 1.1, .1):
                    for alpha in [0.33642712, 0.5808353,  0.74184458, 0.84413189, 0.90962649, 0.94937783,
 0.97398732, 0.98820494, 0.99600274, 1.        ]:
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                                               'extra_weeks_to_simulate':4,
                                               'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                                               'full_activity_alpha':alpha}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                elif experiment_to_run == 'rerun_best_models_and_save_cases_per_poi':
                    # Rerun best fit models so that we can track the infection contribution of each POI,
                    # overall and for each income decile.
                    kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                    simulation_kwargs = {
                        'groups_to_track_num_cases_per_poi':['all',
                            'median_household_income_bottom_decile',
                            'median_household_income_top_decile']}
                    kwarg_copy['model_kwargs']['simulation_kwargs'] = simulation_kwargs
                    kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                    list_of_data_and_model_kwargs.append(kwarg_copy)

            print("In total, it took %2.3f seconds to generate configs for MSA" % (time.time() - msa_t0))

        # sanity check to make sure nothing strange - number of parameters we expect.
        expt_params = []
        for row in list_of_data_and_model_kwargs:
            expt_params.append(
                {'home_beta':row['model_kwargs']['exogenous_model_kwargs']['home_beta'],
                 'poi_psi':row['model_kwargs']['exogenous_model_kwargs']['poi_psi'],
                 'p_sick_at_t0':row['model_kwargs']['exogenous_model_kwargs']['p_sick_at_t0'],
                 'MSA_name':row['data_kwargs']['MSA_name']})
        expt_params = pd.DataFrame(expt_params)

        n_experiments_per_param_setting = expt_params.groupby(['home_beta',
                                                  'poi_psi',
                                                  'p_sick_at_t0',
                                                 'MSA_name']).size()
        if experiment_to_run == 'test_interventions':
            assert (n_experiments_per_param_setting.values == poi_categories_to_examine * 2).all()
        elif experiment_to_run == 'test_max_capacity_clipping':
            assert (n_experiments_per_param_setting.values == 10).all()
        elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
            assert (n_experiments_per_param_setting.values == 10).all()
        elif experiment_to_run == 'rerun_best_models_and_save_cases_per_poi':
            assert (n_experiments_per_param_setting.values == 1).all()
        else:
            assert (n_experiments_per_param_setting.values == 8).all()

    else:  # if experiment_to_run is not in best_models_experiments
        if experiment_to_run != 'just_save_ipf_output':  # model_kwargs is already set for ipf experiment
            model_kwargs = []
            for config in configs_with_changing_params:
                model_kwargs.append({'min_datetime':min_datetime,
                                     'max_datetime':max_datetime,
                                     'exogenous_model_kwargs': {
                                        'home_beta':config['home_beta'],
                                        'poi_psi':config['poi_psi'],
                                        'p_sick_at_t0':config['p_sick_at_t0'],
                                        'just_compute_r0':'calibrate_r0' in experiment_to_run,
                                      },
                                     'model_init_kwargs':{
                                         'num_seeds':num_seeds
                                     },
                                     'simulation_kwargs':{
                                         'use_aggregate_mobility':'aggregate_mobility' in experiment_to_run,
                                     },
                                     'poi_attributes_to_clip':{
                                         'clip_areas':True,
                                         'clip_dwell_times':True,
                                         'clip_visits':True
                                     },
                                     'include_cbg_prop_out':False, # not needed bc we're using preloading IPF output
                                    })

        list_of_data_and_model_kwargs = [{'data_kwargs':copy.deepcopy(a), 'model_kwargs':copy.deepcopy(b), 'experiment_to_run':experiment_to_run} for b in model_kwargs for a in data_kwargs]

    # remove previously fitted kwargs
    if len(previously_fitted_data_and_model_kwargs) > 0:
        print("Prior to filtering out previously fitted kwargs, %i kwargs" % len(list_of_data_and_model_kwargs))
        for i in range(len(previously_fitted_data_and_model_kwargs)):
            # remove stuff that is added when saved so configs are comparable.
            if 'counties_to_track' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
                del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['counties_to_track']
            #if 'preload_poi_visits_list_filename' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
            #    del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['preload_poi_visits_list_filename']

        old_len = len(list_of_data_and_model_kwargs)
        list_of_data_and_model_kwargs = [a for a in list_of_data_and_model_kwargs if a not in previously_fitted_data_and_model_kwargs]
        assert old_len != len(list_of_data_and_model_kwargs)
        print("After removing previously fitted kwargs, %i kwargs" % (len(list_of_data_and_model_kwargs)))

    print("Total data/model configs to fit: %i; randomly shuffling order" % len(list_of_data_and_model_kwargs))
    random.Random(0).shuffle(list_of_data_and_model_kwargs)
    if config_idx_to_start_at is not None:
        print("Skipping first %i configs" % config_idx_to_start_at)
        list_of_data_and_model_kwargs = list_of_data_and_model_kwargs[config_idx_to_start_at:]
    print("Total time to generate configs: %2.3f seconds" % (time.time() - config_generation_start_time))
    return list_of_data_and_model_kwargs


def get_list_of_poi_subcategories_with_most_visits(n_poi_categories, n_chunks=5, return_df_without_filtering_or_sorting=False):
    """
    Remove blacklisted categories and return n_poi_categories subcategories with the most visits in "normal times" (Jan 2019 - Feb 2020)
    """
    normal_times = helper.list_datetimes_in_range(datetime.datetime(2019, 1, 1),
                                              datetime.datetime(2020, 2, 29))
    normal_time_cols = ['%i.%i.%i' % (a.year, a.month, a.day) for a in normal_times]
    must_have_cols = normal_time_cols + ['sub_category', 'top_category']
    d = helper.load_multiple_chunks(range(n_chunks), cols=must_have_cols)
    d['visits_in_normal_times'] = d[normal_time_cols].sum(axis=1)
    if return_df_without_filtering_or_sorting:
        d = d[['sub_category', 'visits_in_normal_times']]
        grouped_d = d.groupby(['sub_category']).agg(['sum', 'size']).reset_index()
        grouped_d.columns = ['Original Name', 'N visits', 'N POIs']
        grouped_d['% POIs'] = 100 * grouped_d['N POIs'] / grouped_d['N POIs'].sum()
        grouped_d['% visits'] = 100 * grouped_d['N visits'] / grouped_d['N visits'].sum()
        grouped_d['Category'] = grouped_d['Original Name'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
        grouped_d = grouped_d.sort_values(by='% visits')[::-1].head(n=n_poi_categories)[['Category', '% visits', '% POIs', 'N visits', 'N POIs']]
        print('Percent of POIs: %2.3f; percent of visits: %2.3f' % 
              (grouped_d['% POIs'].sum(), 
               grouped_d['% visits'].sum()))
        return grouped_d      
    assert all([a in d['sub_category'].values for a in SUBCATEGORY_BLACKLIST])
    assert((d.groupby('sub_category')['top_category'].nunique().values == 1).all()) # Make sure that each sub_category only maps to one top category (and so it's safe to just look at sub categories).
    d = d.loc[d['sub_category'].map(lambda x:x not in SUBCATEGORY_BLACKLIST)]
    grouped_d = d.groupby('sub_category')['visits_in_normal_times'].sum().sort_values()[::-1].iloc[:n_poi_categories]
    print("Returning the list of %i POI subcategories with the most visits, collectively accounting for percentage %2.1f%% of visits" %
        (n_poi_categories, 100*grouped_d.values.sum()/d['visits_in_normal_times'].sum()))
    return list(grouped_d.index)

def filter_timestrings_for_properties(required_properties=None,
                                      required_model_kwargs=None,
                                      required_data_kwargs=None,
                                      min_timestring=None,
                                      max_timestring=None,
                                      return_msa_names=False,
                                      old_directory=False):
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
    if old_directory:
        config_dir = os.path.join(OLD_FITTED_MODEL_DIR, 'data_and_model_configs')
    else:
        config_dir = os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs')
    matched_timestrings = []
    msa_names = []
    configs_to_evaluate = os.listdir(config_dir)
    print("%i files in directory %s" % (len(configs_to_evaluate), config_dir))
    for fn in configs_to_evaluate:
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

def check_memory_usage():
    virtual_memory = psutil.virtual_memory()
    total_memory = getattr(virtual_memory, 'total')
    available_memory = getattr(virtual_memory, 'available')
    free_memory = getattr(virtual_memory, 'free')
    available_memory_percentage = 100. * available_memory / total_memory
    # Free memory is the amount of memory which is currently not used for anything. This number should be small, because memory which is not used is simply wasted.
    # Available memory is the amount of memory which is available for allocation to a new process or to existing processes.
    print('Total memory: %s; free memory: %s; available memory %s; available memory %2.3f%%' % (
        bytes2human(total_memory),
        bytes2human(free_memory),
        bytes2human(available_memory),
        available_memory_percentage))
    return available_memory_percentage

def run_many_models_in_parallel(configs_to_fit):
    max_processes_for_user = int(multiprocessing.cpu_count() / 1.2)
    print("Maximum number of processes to run: %i" % max_processes_for_user)
    for config_idx in range(len(configs_to_fit)):
        t0 = time.time()
        # Check how many processes user is running.
        n_processes_running = int(subprocess.check_output('ps -fA | grep model_experiments.py | wc -l', shell=True))
        print("Current processes running for user: %i" % n_processes_running)
        while n_processes_running > max_processes_for_user:
            print("Current processes are %i, above threshold of %i; waiting." % (n_processes_running, max_processes_for_user))
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
        outfile_path = os.path.join(FITTED_MODEL_DIR, 'model_fitting_logfiles/%s.out' % timestring)
        cmd = 'nohup python -u model_experiments.py fit_and_save_one_model %s --timestring %s --config_idx %i > %s 2>&1 &' % (experiment_to_run, timestring, config_idx, outfile_path)
        print("Command: %s" % cmd)
        os.system(cmd)
        time.sleep(3)
        print("Time between job submissions: %2.3f" % (time.time() - t0))

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
    username = getpass.getuser()
    if username == 'emmap1':
        computers_to_use = ['rambo', 'trinity'] #['furiosa', 'madmax', 'madmax2', 'madmax3', 'madmax4', 'madmax5']
    elif username == 'serinac':
        computers_to_use = ['rambo']
    else:
        raise Exception("Please specify what computers you want to use!")
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

#########################################################
# Functions to evaluate model fit and basic results
#########################################################
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

def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    '''
    Matches the MSA name from our annotated SafeGraph data to the
    MSA name in the external datasource in MSA_COUNTY_MAPPING
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
    acs_data = pd.read_csv(PATH_TO_ACS_5YR_DATA)
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

def resave_fast_to_load_results_for_timestring(ts, old_directory, nyt_outcomes):
    """
    Overwrite old loss if we want to add additional features.
    """
    t0 = time.time()
    model, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(
         ts,
         verbose=False,
         load_fast_results_only=False,
         load_full_model=True,
         old_directory=old_directory)
    model_kwargs = kwargs['model_kwargs']
    data_kwargs = kwargs['data_kwargs']
    train_test_partition = fast_to_load_results['train_test_date_cutoff']
    keys_to_rewrite = ['loss_dict', 'train_loss_dict', 'test_loss_dict']


    for key_to_rewrite in keys_to_rewrite:
        old_loss_dict = None
        new_loss_dict = None
        old_loss_dict = fast_to_load_results[key_to_rewrite]
        if key_to_rewrite == 'loss_dict':
            new_loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                               model_kwargs['min_datetime'],
                                               model=model,
                                               make_plot=False)
        elif key_to_rewrite == 'train_loss_dict':
            train_max = train_test_partition + datetime.timedelta(hours=-1)
            new_loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_end_time = train_max,
                                           model=model)
        elif key_to_rewrite == 'test_loss_dict':
            new_loss_dict = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_start_time = train_test_partition,
                                           model=model)

        common_keys = [a for a in new_loss_dict.keys() if a in old_loss_dict.keys()]
        assert len(common_keys) > 0
        for k in common_keys:
            if type(new_loss_dict[k]) is not np.ndarray:
                assert new_loss_dict[k] == old_loss_dict[k]
            else:
                assert np.allclose(new_loss_dict[k], old_loss_dict[k])

        fast_to_load_results[key_to_rewrite] = new_loss_dict

    if old_directory:
        model_dir = OLD_FITTED_MODEL_DIR
    else:
        model_dir = FITTED_MODEL_DIR
    path_to_save = os.path.join(model_dir, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % ts)
    assert os.path.exists(path_to_save)
    file = open(path_to_save, 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()
    print("Time to save model: %2.3f seconds" % (time.time() - t0))

def compare_model_vs_real_num_cases(nyt_outcomes,
                                    mdl_start_time,
                                    compare_start_time=None,
                                    compare_end_time=None,
                                    model=None,
                                    model_results=None,
                                    mdl_prediction=None,
                                    projected_hrs=None,
                                    detection_rate=.10,
                                    detection_lag=7,
                                    death_rate=.0066,
                                    death_lag=18,
                                    prediction_mode='deterministic',
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
    assert prediction_mode in {'deterministic', 'exponential', 'gamma', 'model_history'}
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
    else:
        mdl_IR = None

    modes = ['cases', 'deaths']

    for mode in modes:

        if mode == 'cases':
            real_data = real_cases
        else:
            real_data = real_deaths

        if not mdl_prediction_provided:
            # note: mdl_prediction should always represent an hourly *cumulative* count per seed x hour
            if mode == 'cases':
                min_thresholds = [1, 10, 20, 50, 100]  # don't evaluate LL on very small numbers -- too noisy
                if prediction_mode == 'deterministic':  # assume constant detection rate and delay
                    mdl_prediction = mdl_IR * detection_rate
                    projected_hrs = [hr + datetime.timedelta(days=detection_lag) for hr in mdl_hours]
                elif prediction_mode == 'exponential':  # draw delays from exponential distribution
                    mdl_hourly_new_cases, _ = draw_cases_and_deaths_from_exponential_distribution(mdl_IR,
                                                detection_rate, detection_lag, death_rate, death_lag)
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
                elif prediction_mode == 'gamma':  # draw delays from gamma distribution
                    mdl_hourly_new_cases, _ = draw_cases_and_deaths_from_gamma_distribution(mdl_IR,
                                                detection_rate, death_rate)
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
                else:  # modeled confirmed cases during simulation
                    assert 'new_confirmed_cases' in history['nyt']
                    mdl_hourly_new_cases = history['nyt']['new_confirmed_cases']
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
            else:
                min_thresholds = [1, 2, 3, 5, 10]  # don't evaluate LL on very small numbers -- too noisy
                if prediction_mode == 'deterministic':  # assume constant detection rate and delay
                    mdl_prediction = mdl_IR * death_rate
                    projected_hrs = [hr + datetime.timedelta(days=death_lag) for hr in mdl_hours]
                elif prediction_mode == 'exponential':  # draw delays from exponential distribution
                    _, mdl_hourly_new_deaths = draw_cases_and_deaths_from_exponential_distribution(mdl_IR,
                                                detection_rate, detection_lag, death_rate, death_lag)
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
                elif prediction_mode == 'gamma':  # draw delays from gamma distribution
                    _, mdl_hourly_new_deaths = draw_cases_and_deaths_from_gamma_distribution(mdl_IR,
                                                detection_rate, death_rate)
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
                else:  # modeled confirmed deaths during simulation
                    assert 'new_confirmed_cases' in history['nyt']
                    mdl_hourly_new_deaths = history['nyt']['new_deaths']
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
            
            if not make_plot:
                # note: y_pred is also cumulative, but represents seed x day, instead of hour
                y_true, y_pred, eval_start, eval_end = find_model_and_real_overlap_for_eval(
                    real_dates, real_data, projected_hrs, mdl_prediction, compare_start_time, compare_end_time)
                if len(y_true) < 5:
                    print("Fewer than 5 days of overlap between model predictions and observed %s data; not scoring" % mode)
                else:
                    score_dict['eval_start_time_%s' % mode] = eval_start
                    score_dict['eval_end_time_%s' % mode] = eval_end
                    score_dict['cumulative_predicted_%s' % mode] = y_pred
                    score_dict['cumulative_true_%s' % mode] = y_true
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
                    # note: all metrics below this should be computed on *daily* cases / deaths per day, not cumulative
                    score_dict['daily_%s_RMSE' % mode] = compute_loss(y_true, y_pred, metric='RMSE', min_threshold=None, compare_daily_not_cumulative=True)
                    score_dict['daily_%s_MSE' % mode] = compute_loss(y_true, y_pred, metric='MSE', min_threshold=None, compare_daily_not_cumulative=True)

                    if prediction_mode == 'deterministic':  # LL metrics assume constant delay and rate for predictions
                        threshold_metrics = [
                            'MRE',
                            'gaussianish_negative_ll',
                            'NLL_with_var_y',
                            'NLL_with_var_ysq',
                            'poisson_NLL']
                        rate = detection_rate if mode == 'cases' else death_rate
                        for threshold_metric in threshold_metrics:
                            for min_threshold in min_thresholds:
                                for do_logsumexp in [True, False]:
                                    if do_logsumexp:
                                        agg_str = 'logsumexp'
                                    else:
                                        agg_str = 'sum'

                                    # Skip logsumexp for MRE since it has no LL interpretation
                                    if threshold_metric == 'MRE' and do_logsumexp:
                                        continue

                                    dict_str = f'daily_{mode}_{threshold_metric}_thres-{min_threshold}_{agg_str}'
                                    score_dict[dict_str] = compute_loss(
                                        y_true=y_true,
                                        y_pred=y_pred,
                                        rate=rate,
                                        metric=threshold_metric,
                                        min_threshold=min_threshold,
                                        compare_daily_not_cumulative=True,
                                        do_logsumexp=do_logsumexp)

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
                if plot_daily_not_cumulative and not smooth_daily_cases_when_plotting:
                    # use non-connected x's if plotting non-smoothed daily cases / deaths
                    ax.plot_date(real_dates, real_data, label=true_line_label, marker='x', c=real_data_color, markersize=marker_size+1, markeredgewidth=2)
                else:
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
                ax.legend(fontsize=legend_fontsize, loc='upper left')

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
                    # Round to nearest hundred
                    top = (top // 100) * 100
                elif plot_mode == 'deaths':
                    # Round to nearest 20
                    top = (top // 20) * 20
                ax.set_yticks([bot, top])

            if plot_mode == 'cases':
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '{:.1f}'.format(x/1000) + 'k'))

            ax.grid(alpha=.5)
            ax.set_title(title, fontsize=title_fontsize)

    return score_dict

def draw_cases_and_deaths_from_exponential_distribution(model_IR, detection_rate, detection_lag_in_days,
                                                        death_rate, death_lag_in_days, random_seed=0):
    # model_IR should be a matrix of seed x hour, where each entry represents the *cumulative* number
    # of people in infectious or removed for that seed and hour
    # eg mdl_IR = (model.history['nyt']['infected'] + model.history['nyt']['removed'])
#     print('Making stochastic predictions: drawing number of days for onset-to-reporting from Exp(%.3f) and onset-to-death from Exp(%.3f)' % (detection_lag_in_days, death_lag_in_days))
    np.random.seed(random_seed)
    detection_lag = detection_lag_in_days * 24  # want the lags in hours
    death_lag = death_lag_in_days * 24
    num_seeds, num_hours = model_IR.shape
    assert num_hours % 24 == 0
    hourly_new_infectious = get_daily_from_cumulative(model_IR)

    predicted_cases = np.zeros((num_seeds, num_hours))
    predicted_deaths = np.zeros((num_seeds, num_hours))
    cases_to_confirm = np.zeros(num_seeds)
    deaths_to_happen = np.zeros(num_seeds)
    for hr in range(num_hours):
        new_infectious = hourly_new_infectious[:, hr]
        new_confirmed_cases = np.random.binomial(cases_to_confirm.astype(int), 1/detection_lag)
        predicted_cases[:, hr] = new_confirmed_cases
        new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), detection_rate)
        cases_to_confirm = cases_to_confirm + new_cases_to_confirm - new_confirmed_cases
        new_deaths = np.random.binomial(deaths_to_happen.astype(int), 1/death_lag)
        predicted_deaths[:, hr] = new_deaths
        new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), death_rate)
        deaths_to_happen = deaths_to_happen + new_deaths_to_happen - new_deaths
    return predicted_cases, predicted_deaths

def draw_cases_and_deaths_from_gamma_distribution(model_IR, detection_rate, death_rate,
                                                  detection_delay_shape=1.85,  # Li et al. (Science 2020)
                                                  detection_delay_scale=3.57,
                                                  death_delay_shape=1.85,
                                                  death_delay_scale=9.72,
                                                  random_seed=0):
    # model_IR should be a matrix of seed x hour, where each entry represents the *cumulative* number
    # of people in infectious or removed for that seed and hour
    # eg mdl_IR = (model.history['nyt']['infected'] + model.history['nyt']['removed'])
#     print('Making stochastic predictions: drawing number of days for onset-to-reporting from Gamma(%.3f, %.3f) [mean=%.3f] and onset-to-death from Gamma(%.3f, %.3f) [mean=%.3f]' %
#           (detection_delay_shape, detection_delay_scale, detection_delay_shape*detection_delay_scale,
#            death_delay_shape, death_delay_scale, death_delay_shape*death_delay_scale))
    np.random.seed(random_seed)
    num_seeds, num_hours = model_IR.shape
    assert num_hours % 24 == 0
    hourly_new_infectious = get_daily_from_cumulative(model_IR)

    predicted_cases = np.zeros((num_seeds, num_hours))
    predicted_deaths = np.zeros((num_seeds, num_hours))
    for hr in range(num_hours):
        new_infectious = hourly_new_infectious[:, hr]  # 1 x S
        cases_to_confirm = np.random.binomial(new_infectious.astype(int), detection_rate)
        deaths_to_happen = np.random.binomial(new_infectious.astype(int), death_rate)
        for seed in range(num_seeds):
            num_cases = cases_to_confirm[seed]
            confirmation_delays = np.random.gamma(detection_delay_shape, detection_delay_scale, int(num_cases))
            confirmation_delays = confirmation_delays * 24  # convert delays from days to hours
            counts = Counter(confirmation_delays).most_common()
            for delay, count in counts:
                projected_hr = int(hr + delay)
                if projected_hr < num_hours:
                    predicted_cases[seed, projected_hr] = predicted_cases[seed, projected_hr] + count

            num_deaths = deaths_to_happen[seed]
            death_delays = np.random.gamma(death_delay_shape, death_delay_scale, int(num_deaths))
            death_delays = death_delays * 24  # convert delays from days to hours
            counts = Counter(death_delays).most_common()
            for delay, count in counts:
                projected_hr = int(hr + delay)
                if projected_hr < num_hours:
                    predicted_deaths[seed, projected_hr] = predicted_deaths[seed, projected_hr] + count
    return predicted_cases, predicted_deaths

def compute_loss(y_true, y_pred, rate=None,
                 metric='RMSE',
                 min_threshold=None,
                 compare_daily_not_cumulative=True,
                 do_logsumexp=False):
    """
    This assumes that y_true and y_pred are cumulative counts.
    y_true: 1D array, the true case/death counts
    y_pred: 2D array, the predicted case/death counts over all seeds
    rate: the detection or death rate used in computing y_pred;
          only required when metric is poisson_NLL or binomial_NLL
    metric: RMSE or MRE, the loss metric
    min_threshold: the minimum number of true case/deaths that a day must have
                   to be included in eval
    compare_daily_not_cumulative: converts y_true and y_pred into daily counts
                                  and does the comparison on those instead
    do_logsumexp: whether to sum or logsumexp over seeds for LL metrics
    """
    assert metric in {
        'RMSE',
        'MRE',
        'MSE',
        'gaussianish_negative_ll',
        'NLL_with_var_y',
        'NLL_with_var_ysq',
        'poisson_NLL',
        'binomial_NLL'}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if compare_daily_not_cumulative:
        y_true = get_daily_from_cumulative(y_true)
        y_pred = get_daily_from_cumulative(y_pred)
    else:
        assert metric not in [
            'gaussianish_negative_ll',
            'NLL_with_var_y',
            'NLL_with_var_ysq',
            'poisson_NLL',
            'binomial_NLL']

    if do_logsumexp:
        sum_or_logsumexp = logsumexp
    else:
        sum_or_logsumexp = np.sum

    if min_threshold is not None:
        orig_len = len(y_true)
        idxs = y_true >= min_threshold
        if not idxs.sum() > 0:
            print(y_true)
            print("Warning: NOT ENOUGH VALUES ABOVE THRESHOLD %s" % min_threshold)
            return np.nan
        y_true = y_true[idxs]
        y_pred = y_pred[:, idxs]
        num_dropped = orig_len - len(y_true)
        if num_dropped > 30:
            print('Warning: dropped %d dates after applying min_threshold %d' % (num_dropped, min_threshold))

    if metric == 'RMSE':
        return RMSE(y_true=y_true, y_pred=y_pred)
    elif metric == 'MRE':
        return MRE(y_true=y_true, y_pred=y_pred)
    elif metric == 'MSE':
        return MSE(y_true=y_true, y_pred=y_pred)
    elif metric == 'gaussianish_negative_ll':
        return gaussianish_negative_ll(
            y_true=y_true,
            y_pred=y_pred,
            sum_or_logsumexp=sum_or_logsumexp)
    elif metric == 'NLL_with_var_y':
        return NLL_with_var_y(
            y_true=y_true,
            y_pred=y_pred,
            sum_or_logsumexp=sum_or_logsumexp)
    elif metric == 'NLL_with_var_ysq':
        return NLL_with_var_ysq(
            y_true=y_true,
            y_pred=y_pred,
            sum_or_logsumexp=sum_or_logsumexp)
    elif metric == 'poisson_NLL':
        return poisson_NLL(
            y_true=y_true,
            y_pred=y_pred,
            sum_or_logsumexp=sum_or_logsumexp)
    elif metric == 'binomial_NLL':
        assert rate is not None
        return binomial_NLL(
            y_true=y_true,
            y_pred=y_pred,
            rate=rate,
            sum_or_logsumexp=sum_or_logsumexp)

def evaluate_all_fitted_models_for_msa(msa_name, min_timestring=None,
                                        max_timestring=None,
                                        timestrings=None,
                                       required_properties=None,
                                       required_model_kwargs=None,
                                       recompute_losses=False,
                                       key_to_sort_by=None,
                                       old_directory=False):

    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs`
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
            max_timestring=max_timestring,
            old_directory=old_directory)
        print('Found %d fitted models for %s' % (len(timestrings), msa_name))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    if recompute_losses:
        nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)

    results = []
    start_time = time.time()
    for ts in timestrings:
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False,
            load_fast_results_only=(not recompute_losses), old_directory=old_directory)
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

def evaluate_all_fitted_models_for_experiment(experiment_to_run,
                                              min_timestring=None,
                                              max_timestring=None,
                                              timestrings=None,
                                              required_properties=None,
                                              required_model_kwargs=None,
                                              required_data_kwargs=None,
                                              result_types=None,
                                              key_to_sort_by=None,
                                              old_directory=False):

    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs`
    """
    if required_properties is None:
        required_properties = {}
    required_properties['experiment_to_run'] = experiment_to_run
    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_data_kwargs is None:
        required_data_kwargs = {}

    if timestrings is None:
        timestrings = filter_timestrings_for_properties(
            required_properties=required_properties,
            required_model_kwargs=required_model_kwargs,
            required_data_kwargs=required_data_kwargs,
            min_timestring=min_timestring,
            max_timestring=max_timestring,
            old_directory=old_directory)
        print('Found %d fitted models for %s' % (len(timestrings), experiment_to_run))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    if result_types is None:
        result_types = ['loss_dict', 'train_loss_dict', 'test_loss_dict']
        # ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']
    results = []
    start_time = time.time()
    for i, ts in enumerate(timestrings):
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False, load_fast_results_only=True, old_directory=old_directory)
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']

        results_for_ts = {'timestring':ts,
                         'data_kwargs':data_kwargs,
                         'model_kwargs':model_kwargs,
                         'results':model_results,
                         'experiment_to_run':experiment_to_run}

        if 'final infected fraction' in fast_to_load_results:
            results_for_ts['final infected fraction'] = fast_to_load_results['final infected fraction']

        for result_type in result_types:
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
        if i % 1000 == 0:
            curr_time = time.time()
            print('Loaded %d models so far: %.3fs -> %.3fs per model' %
                  (len(results), curr_time-start_time, (curr_time-start_time)/len(results)))

    end_time = time.time()
    print('Time to load and score all models: %.3fs -> %.3fs per model' %
          (end_time-start_time, (end_time-start_time)/len(timestrings)))
    results = pd.DataFrame(results)

    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results

if __name__ == '__main__':
    # command line arguments.
    # Basically, this script can be called two ways: either as a manager job, which generates configs and fires off a bunch of worker jobs
    # or as a worker job, which runs a single model with a single config.
    # The command line argument manager_or_worker_job specifies which of these two usages we're using.
    # The other important command line argument is experiment_to_run, which specifies which step of the experimental pipeline we're running.
    # The worker jobs take additional arguments like timestring (which specifies the timestring we use to save model files)
    # and config_idx, which specifies which config we're using.
    valid_experiments = ['normal_grid_search', 'calibrate_r0',
                 'grid_search_aggregate_mobility', 'calibrate_r0_aggregate_mobility',
                 'just_save_ipf_output', 'test_interventions',
                 'test_retrospective_counterfactuals', 'test_max_capacity_clipping',
                 'test_uniform_proportion_of_full_reopening', 'rerun_best_models_and_save_cases_per_poi']
    parser = argparse.ArgumentParser()
    parser.add_argument('manager_or_worker_job', help='Is this the manager job or the worker job?',
        choices=['run_many_models_in_parallel', 'fit_and_save_one_model'])
    parser.add_argument('experiment_to_run', help='The name of the experiment to run')
    parser.add_argument('--timestring', type=str)
    parser.add_argument('--config_idx', type=int)
    parser.add_argument('--how_to_select_best_grid_search_models', type=str, choices=['daily_cases_rmse', 'daily_deaths_rmse', 'daily_cases_poisson'])
    args = parser.parse_args()

    # Less frequently used arguments.
    config_idx_to_start_at = None
    skip_previously_fitted_kwargs = False
    min_timestring = '2020_07_16_10_4'
    min_timestring_to_load_best_fit_models_from_grid_search = '2020_07_16_10_4' # what grid search models do we look at when loading interventions.

    config_filename = '%s_configs.pkl' % COMPUTER_WE_ARE_RUNNING_ON.replace('.stanford.edu', '')
    if args.manager_or_worker_job == 'run_many_models_in_parallel':
        # manager job generates configs.
        assert args.timestring is None
        assert args.config_idx is None
        experiment_list = args.experiment_to_run.split(',')
        assert [a in valid_experiments for a in experiment_list]
        print("Starting the following list of experiments")
        print(experiment_list)
        configs_to_fit = []
        for experiment in experiment_list:
            if experiment not in ['normal_grid_search', 'calibrate_r0', 'grid_search_aggregate_mobility', 'just_save_ipf_output']:
                assert args.how_to_select_best_grid_search_models is not None, 'Error: must specify how you wish to select best-fit models'
            configs_for_experiment = generate_data_and_model_configs(config_idx_to_start_at=config_idx_to_start_at,
                skip_previously_fitted_kwargs=skip_previously_fitted_kwargs,
                min_timestring=min_timestring,
                experiment_to_run=experiment,
                how_to_select_best_grid_search_models=args.how_to_select_best_grid_search_models,
                min_timestring_to_load_best_fit_models_from_grid_search=min_timestring_to_load_best_fit_models_from_grid_search)
            configs_for_experiment = partition_jobs_across_computers(COMPUTER_WE_ARE_RUNNING_ON, configs_for_experiment)
            configs_to_fit += configs_for_experiment
        print("Total number of configs to run on %s (%i experiments): %i" % (COMPUTER_WE_ARE_RUNNING_ON, len(configs_to_fit), len(experiment_list)))
        f = open(config_filename, 'wb')
        pickle.dump(configs_to_fit, f)
        f.close()
    else:
        # worker job needs to load the list of configs and figure out which one it's running.
        assert args.experiment_to_run in valid_experiments
        print("loading configs from %s" % config_filename)
        f = open(config_filename, 'rb')
        configs_to_fit = pickle.load(f)
        f.close()



    if args.manager_or_worker_job == 'run_many_models_in_parallel':
        # fire off worker jobs.
        run_many_models_in_parallel(configs_to_fit)
    elif args.manager_or_worker_job == 'fit_and_save_one_model':
        # single worker job; use command line arguments to retrieve config and timestring.
        timestring = args.timestring
        config_idx = args.config_idx
        assert timestring is not None and config_idx is not None
        data_and_model_config = configs_to_fit[config_idx]

        if 'grid_search' in args.experiment_to_run:
            train_test_partition = TRAIN_TEST_PARTITION
        else:
            train_test_partition = None

        print("Running single model. Kwargs are")
        print_config_as_json(data_and_model_config)

        fit_and_save_one_model(timestring,
            train_test_partition=train_test_partition,
            model_kwargs=data_and_model_config['model_kwargs'],
            data_kwargs=data_and_model_config['data_kwargs'],
            experiment_to_run=data_and_model_config['experiment_to_run'])
    else:
        raise Exception("This is not a valid way to call this method")
