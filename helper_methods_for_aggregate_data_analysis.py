from covid_constants_and_util import *
import geopandas as gpd
import statsmodels.api as sm
import json
import copy
from fbprophet import Prophet
from collections import Counter
import re
import h5py
import ast
from shapely import wkt
from scipy.stats import pearsonr
import csv
import os

# define some paths
STRATIFIED_BY_AREA_DIR = '/dfs/scratch1/safegraph_homes/all_aggregate_data/chunks_with_demographic_annotations_stratified_by_area/'
H5_DATA_DIR = '/dfs/scratch1/safegraph_homes/all_aggregate_data/h5data/'
ANNOTATED_H5_DATA_DIR = '/dfs/scratch1/safegraph_homes/all_aggregate_data/chunks_with_demographic_annotations/'
CHUNK_FILENAME = 'chunk_1.2017-3.2020_c2.h5'
FITTED_MODEL_DIR = '/dfs/scratch1/safegraph_homes/all_aggregate_data/fitted_models/'
UNZIPPED_DATA_DIR = '/dfs/scratch1/safegraph_homes/all_aggregate_data/20191213-safegraph-aggregate-longitudinal-data-to-unzip-to/'
ACS_DIR = '/dfs/scratch1/safegraph_homes/ACS_17_5YR_tables/'
PATH_TO_ACS_5YR_DATA = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/2017_five_year_acs_data/2017_five_year_acs_data.csv'
PATH_TO_ACS_1YR_DATA = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/2018_one_year_acs_population_data/nhgis0001_ds239_20185_2018_blck_grp.csv'
MSA_COUNTY_MAPPING = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/msa_county_mapping.csv'
PATH_TO_GOOGLE_DATA = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/20200508_google_mobility_report.csv'

# automatically read weekly strings so we don't have to remember to update it each week.
ALL_WEEKLY_STRINGS = sorted([a.replace('-weekly-patterns.csv.gz', '') for a in os.listdir('/dfs/scratch1/safegraph_homes/all_aggregate_data/weekly_patterns_data/v1/main-file/')])
try:
    cast_to_datetime = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in ALL_WEEKLY_STRINGS]
except:
    print(ALL_WEEKLY_STRINGS)
    raise Exception("At least one weekly string is badly formatted.")

def load_social_distancing_metrics(datetimes, version='v2'):
    """
    Given a list of datetimes, load social distancing metrics for those days.

    load_social_distancing_metrics(helper.list_datetimes_in_range(datetime.datetime(2020, 3, 1),
                                                                  datetime.datetime(2020, 3, 7)))
    """
    print("Loading social distancing metrics for %i datetimes; using version %s" % (len(datetimes), version))
    t0 = time.time()
    daily_cols = ['device_count', 'distance_traveled_from_home',
                    'completely_home_device_count', 'full_time_work_behavior_devices']
    concatenated_d = None
    for datetime in datetimes:
        if version == 'v1':
            path = os.path.join('/dfs/scratch1/safegraph_homes/all_aggregate_data/daily_counts_of_people_leaving_homes/sg-social-distancing/',
                            datetime.strftime('%Y/%m/%d/%Y-%m-%d-social-distancing.csv.gz'))
        elif version == 'v2':
            path = os.path.join('/dfs/scratch1/safegraph_homes/all_aggregate_data/daily_counts_of_people_leaving_homes/social_distancing_v2/',
                            datetime.strftime('%Y/%m/%d/%Y-%m-%d-social-distancing.csv.gz'))
        else:
            raise Exception("Version should be v1 or v2")
        #print("Loading file from %s" % path)

        if os.path.exists(path):
            social_distancing_d = pd.read_csv(path, usecols=['origin_census_block_group'] + daily_cols)
            social_distancing_d.columns = ['census_block_group'] + ['%i.%i.%i_%s' %
                                                                    (datetime.year, datetime.month, datetime.day, a) for a in daily_cols]
            old_len = len(social_distancing_d)
            social_distancing_d = social_distancing_d.drop_duplicates(keep=False)
            n_dropped_rows = old_len - len(social_distancing_d)
            assert len(set(social_distancing_d['census_block_group'])) == len(social_distancing_d)
            assert(1.*n_dropped_rows/old_len < 0.002) # make sure not very many rows are duplicates.
            if version == 'v2':
                assert n_dropped_rows == 0 # they fixed the problem in v2.
            elif version == 'v1':
                assert n_dropped_rows > 0 # this seemed to be a problem in v1.

            if concatenated_d is None:
                concatenated_d = social_distancing_d
            else:
                concatenated_d = pd.merge(concatenated_d,
                                          social_distancing_d,
                                          how='outer',
                                          validate='one_to_one',
                                          on='census_block_group')
        else:
            print('Missing Social Distancing Metrics for %s' % datetime.strftime('%Y/%m/%d'))
    if concatenated_d is None:  # could not find any of the dates
        return concatenated_d
    print("Total time to load social distancing metrics: %2.3f seconds; total rows %i" %
          (time.time() - t0, len(concatenated_d)))
    return concatenated_d

def annotate_with_demographic_info_and_write_out_in_chunks(full_df, just_testing=False):
    """
    Full df
    """
    full_df['safegraph_place_id'] = full_df.index
    full_df.index = range(len(full_df))

    # merge with areas.
    safegraph_areas = pd.read_csv('/dfs/scratch1/safegraph_homes/all_aggregate_data/safegraph_poi_area_calculations/SafeGraphPlacesGeoSupplementSquareFeet.csv.gz')
    print("Prior to merging with safegraph areas, %i rows" % len(full_df))
    safegraph_areas = safegraph_areas[['safegraph_place_id', 'area_square_feet']].dropna()
    safegraph_areas.columns = ['safegraph_place_id', 'safegraph_computed_area_in_square_feet']
    full_df = pd.merge(full_df, safegraph_areas, how='inner', on='safegraph_place_id', validate='one_to_one')
    print("After merging with areas, %i rows" % len(full_df))

    # map to demo info.
    print("Mapping SafeGraph POIs to demographic info, including race, income, and vote share.")
    gdb_files = ['ACS_2017_5YR_BG_51_VIRGINIA.gdb'] if just_testing else None
    cbg_mapper = dataprocessor.CensusBlockGroups(base_directory='/dfs/scratch1/safegraph_homes/old_dfs_scratch0_directory_contents/new_census_data/',
                          gdb_files=gdb_files)#)#)
    pop_df = load_dataframe_to_correct_for_population_size()


    chunksize = 100000

    for chunk_number in range(len(full_df) // chunksize + 1):
        print("******************Annotating chunk %i" % chunk_number)
        start, end = chunk_number * chunksize, min((chunk_number + 1) * chunksize, len(full_df))
        #.to_hdf(os.path.join(helper.H5_DATA_DIR, helper.CHUNK_FILENAME), f'chunk_{i}', mode='a', complevel=2)
        d = full_df.iloc[start:end].copy()

        #d = load_chunk(chunk_number, load_with_annotations=False, load_backup=just_testing)



        # Now annotate each POI on the basis of its location.
        election_results_data = load_election_results_data()

        mapped_pois = cbg_mapper.get_demographic_stats_of_points(d['latitude'].values,
                                          d['longitude'].values,
                                          desired_cols=['p_white', 'p_asian', 'p_black', 'median_household_income', 'people_per_mile'])
        mapped_pois['county_fips_code'] = mapped_pois['county_fips_code'].map(lambda x:int(x) if x is not None else x)
        mapped_pois['state'] = d['region'].values
        old_len = len(mapped_pois)
        mapped_pois = pd.merge(mapped_pois,
                               election_results_data,
                               on=['state', 'county_fips_code'],
                               how='left',
                               validate='many_to_one')
        del mapped_pois['state']
        assert old_len == len(mapped_pois)
        mapped_pois.columns = ['poi_lat_lon_%s' % a for a in mapped_pois.columns]
        for c in mapped_pois.columns:
            d[c] = mapped_pois[c].values

        # Then annotate with demographic data based on where
        #  visitors come from (visitor_home_cbgs).
        d = aggregate_visitor_home_cbgs_over_months(d, population_df=pop_df)
        block_group_d = cbg_mapper.block_group_d.copy()
        block_group_d['id_to_match_to_safegraph_data'] = block_group_d['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
        block_group_d = block_group_d[['id_to_match_to_safegraph_data', 'p_black', 'p_white', 'p_asian', 'median_household_income']]
        block_group_d = block_group_d.dropna()

        for col in block_group_d:
            if col == 'id_to_match_to_safegraph_data':
                continue
            cbg_dict = dict(zip(block_group_d['id_to_match_to_safegraph_data'].values, block_group_d[col].values))
            d['cbg_visitor_weighted_%s' % col] = d['aggregated_visitor_home_cbgs'].map(lambda x:compute_weighted_mean_of_cbg_visitors(x, cbg_dict))


        # see how well we did.
        for c in [a for a in d.columns if 'poi_lat_lon_' in a or 'cbg_visitor_weighted' in a]:
            print("Have data for %s for fraction %2.3f of people" % (c, 1 - pd.isnull(d[c]).mean()))
        pd.set_option('max_rows', 500)
        d.to_hdf(os.path.join(ANNOTATED_H5_DATA_DIR, CHUNK_FILENAME) ,f'chunk_{chunk_number}', mode='a', complevel=2)

def load_date_col_as_date(x):
    try:
        year, month, day = x.split('.')
        if (int(year) in range(2010, 2030)) and (int(month) in range(1, 13)) and (int(day) in range(1, 32)):
            return datetime.datetime(int(year), int(month), int(day))
    except:
        return None

def get_h5_filepath(load_with_annotations, load_backup):
    assert load_with_annotations # we are no longer saving non-annotated files.
    backup_string = 'BACKUP_' if load_backup else ''
    if load_with_annotations:
        filepath = os.path.join(ANNOTATED_H5_DATA_DIR, backup_string + CHUNK_FILENAME)
    else:
        filepath = os.path.join(H5_DATA_DIR, backup_string + CHUNK_FILENAME)
    return filepath

def load_chunk(chunk, load_with_annotations=True, load_backup=False):
    """
    Load a single 100k chunk from the h5 file; chunks are randomized and so should be reasonably
    representative. Currently takes about 12 secs to load a chunk (on madmax) without loading demographic info.
    """
    filepath = get_h5_filepath(load_with_annotations=load_with_annotations, load_backup=load_backup)
    print("Reading chunk %i from %s" % (chunk, filepath))

    d = pd.read_hdf(filepath, key=f'chunk_{chunk}')
    date_cols = [load_date_col_as_date(a) for a in d.columns]
    date_cols = [a for a in date_cols if a is not None]
    print("Dates range from %s to %s" % (min(date_cols), max(date_cols)))
    return d

def load_multiple_chunks(chunks, load_with_annotations=True, load_backup=False, cols=None):
    """
    Loads multiple chunks from the h5 file. Currently quite slow; quicker if only a subset of columns are kept.
    Use the parameters cols to specify which columns to keep; if None then all are kept.
    """
    dfs = []
    for i in chunks:
        t0 = time.time()
        chunk = load_chunk(i, load_with_annotations=load_with_annotations, load_backup=load_backup)
        print("Loaded chunk %i in %2.3f seconds" % (i, time.time() - t0))
        if cols is not None:
            chunk = chunk[cols]
        dfs.append(chunk)
    t0 = time.time()
    df = pd.concat(dfs)
    print("Concatenated %d chunks in %2.3f seconds" % (len(chunks), time.time() - t0))
    return df

def load_all_chunks(cols=None, load_backup=False, load_with_annotations=True):
    """
    Load all 100k chunks from the h5 file. This currently takes a while.
    """
    filepath = get_h5_filepath(load_with_annotations=load_with_annotations, load_backup=load_backup)
    f = h5py.File(filepath, 'r')
    chunks = sorted([int(a.replace('chunk_', '')) for a in list(f.keys())])
    f.close()
    assert chunks == list(range(max(chunks) + 1))
    print("Loading all chunks: %s" % (','.join([str(a) for a in chunks])))
    return load_multiple_chunks(chunks, cols=cols, load_backup=load_backup, load_with_annotations=load_with_annotations)

def unzip_all_aggregate_data_files():
    """
    Unzip the original files downloaded from SafeGraph and put in another folder.
    May need to change paths as you get the new data.
    """
    raise Exception("You likely don't want to do this again without clearing out the new directory first. Only need to perform this step if you get new data.")
    original_dir = '/dfs/scratch1/safegraph_homes/all_aggregate_data/20200316-safegraph-aggregate-longitudinal-data-update'
    zipped_files = sorted(os.listdir(original_dir))
    assert all([f.endswith('.zip') for f in zipped_files])
    for idx, filename in enumerate(zipped_files):
        print("Unzipping file %i/%i" % (idx + 1, len(zipped_files)))
        old_filepath = os.path.join(original_dir, filename)

        new_filepath = os.path.join(UNZIPPED_DATA_DIR, filename.replace('.zip', ''))
        cmd = 'unzip %s -d %s' % (old_filepath, new_filepath)
        print(cmd)
        os.system(cmd)

def load_patterns_data(month=None, year=None, week_string=None, extra_cols=[], just_testing=False):
    """
    Load in data for a single month and year, or for a single week.
    Use extra_cols to define non-default columns to load.
    """
    change_by_date = ['visitor_home_cbgs', 'visitor_country_of_origin',
                      'distance_from_home', 'median_dwell', 'bucketed_dwell_times']

    if month is not None and year is not None:
        month_and_year = True
        assert week_string is None
        assert month in range(1, 13)
        assert year in [2017, 2018, 2019, 2020]
        if (year == 2019 and month == 12) or (year == 2020 and month in [1, 2]):
            upload_date_string = '2020-03-16'  # we uploaded files in two groups; load them in the same way.
        else:
            upload_date_string = '2019-12-12'
        month_and_year_string = '%i_%02d-%s' % (year, month, upload_date_string)
        base_dir = os.path.join(UNZIPPED_DATA_DIR, 'SearchofAllRecords-CORE_POI-GEOMETRY-PATTERNS-%s' % month_and_year_string)
        print("Loading all files from %s" % base_dir)

        filenames = [a for a in os.listdir(base_dir) if
                     (a.startswith('core_poi-geometry-patterns-part') and a.endswith('.csv.gz'))]
        if just_testing:
            filenames = filenames[:2]
        print("Number of files to load: %i" % len(filenames))
        full_paths = [os.path.join(base_dir, a) for a in filenames]
        x = load_csv_possibly_with_dask(full_paths, use_dask=True, usecols=['safegraph_place_id',
                                                                            'location_name',
                                                                            'latitude',
                                                                            'longitude',
                                                                            'city',
                                                                            'region',
                                                                            'postal_code',
                                                                            'top_category',
                                                                            'sub_category',
                                                                            'naics_code',
                                                                            "polygon_wkt",
                                                                            "polygon_class",
                                                                            'visits_by_day',
                                                                            'visitor_home_cbgs',
                                                                            'visitor_country_of_origin',
                                                                            'distance_from_home',
                                                                            'median_dwell',
                                                                            'bucketed_dwell_times'] +
                                                                            extra_cols,
                                                                            dtype={'naics_code': 'float64'})
        print("Fraction %2.3f of NAICS codes are missing" % pd.isnull(x['naics_code']).mean())
        x = x.rename(columns={k: f'{year}.{month}.{k}' for k in change_by_date})
    else:
        month_and_year = False
        assert month is None and year is None
        assert week_string in ALL_WEEKLY_STRINGS
        filepath = os.path.join('/dfs/scratch1/safegraph_homes/all_aggregate_data/weekly_patterns_data/v1/main-file/%s-weekly-patterns.csv.gz' % week_string)
        # Filename is misleading - it is really a zipped file.
        # Also, we're missing some columns that we had before, so I think we're just going to have to join on SafeGraph ID.
        x = pd.read_csv(filepath, escapechar='\\' ,compression='gzip', nrows=10000 if just_testing else None, usecols=['safegraph_place_id',
            'visits_by_day',
            'visitor_home_cbgs',
            'visitor_country_of_origin',
            'distance_from_home',
            'median_dwell',
            'bucketed_dwell_times',
            'date_range_start',
            'visits_by_each_hour'])
        x['offset_from_gmt'] = x['date_range_start'].map(lambda x:x.split('-')[-1])
        print("Offset from GMT value counts")
        print(x['offset_from_gmt'].value_counts())
        del x['date_range_start']
        x = x.rename(columns={k: f'{week_string}.{k}' for k in change_by_date})



    print("Prior to dropping rows with no visits by day, %i rows" % len(x))
    x = x.dropna(subset=['visits_by_day'])
    x['visits_by_day'] = x['visits_by_day'].map(json.loads) # convert string lists to lists.

    if month_and_year:
        days = pd.DataFrame(x['visits_by_day'].values.tolist(),
                     columns=[f'{year}.{month}.{day}'
                              for day in range(1, len(x.iloc[0]['visits_by_day']) + 1)])
    else:
        year = int(week_string.split('-')[0])
        month = int(week_string.split('-')[1])
        start_day = int(week_string.split('-')[2])
        start_datetime = datetime.datetime(year, month, start_day)
        all_datetimes = [start_datetime + datetime.timedelta(days=i) for i in range(7)]
        days = pd.DataFrame(x['visits_by_day'].values.tolist(),
                     columns=['%i.%i.%i' % (dt.year, dt.month, dt.day) for dt in all_datetimes])

        # Load hourly data as well.
        # Per SafeGraph documentation:
        # Start time for measurement period in ISO 8601 format of YYYY-MM-DDTHH:mm:SS±hh:mm
        # (local time with offset from GMT). The start time will be 12 a.m. Sunday in local time.
        x['visits_by_each_hour'] = x['visits_by_each_hour'].map(json.loads) # convert string lists to lists.
        assert all_datetimes[0].strftime('%A') == 'Sunday'
        hours = pd.DataFrame(x['visits_by_each_hour'].values.tolist(),
                     columns=[f'hourly_visits_%i.%i.%i.%i' % (dt.year, dt.month, dt.day, hour)
                              for dt in all_datetimes
                              for hour in range(0, 24)])

    days.index = x.index
    x = pd.concat([x, days], axis=1)
    if not month_and_year:
        x = pd.concat([x, hours], axis=1)
        x = x.drop(columns=['visits_by_each_hour'])

        # Annoying: The hourly data has some spurious spikes
        # related to the GMT-day boundary which we have to correct for.
        date_cols = [load_date_col_as_date(a) for a in x.columns]
        date_cols = [a for a in date_cols if a is not None]
        assert len(date_cols) == 7

        if week_string >= '2020-03-15': # think this is because of DST.
            hourly_offsets = [4, 5, 6, 7]
        else:
            hourly_offsets = [5, 6, 7, 8]
        hourly_offset_strings = ['0%i:00' % hourly_offset for hourly_offset in hourly_offsets]

        percent_rows_being_corrected = (x['offset_from_gmt'].map(lambda a:a in hourly_offset_strings).mean() * 100)
        print("%2.3f%% of rows have timezones that we spike-correct for." % percent_rows_being_corrected)
        assert percent_rows_being_corrected > 99

        # have to correct for each timezone separately.
        for hourly_offset in hourly_offsets:
            idxs = x['offset_from_gmt'] == ('0%i:00' % hourly_offset)
            for date_col in date_cols: # loop over days.
                date_string = '%i.%i.%i' % (date_col.year, date_col.month, date_col.day)
                # not totally clear which hours are messed up - it's mainly one hour, but the surrounding ones look weird too -
                # or what the best way to interpolate is, but this yields plots which look reasonable.

                for hour_to_correct in [24 - hourly_offset - 1,
                                        24 - hourly_offset,
                                        24 - hourly_offset + 1]:

                    # interpolate using hours fairly far from hour_to_correct to avoid pollution.
                    if hour_to_correct < 21:
                        cols_to_use = ['hourly_visits_%s.%i' % (date_string, a) for a in [hour_to_correct - 3, hour_to_correct + 3]]
                    else:
                        cols_to_use = ['hourly_visits_%s.%i' % (date_string, a) for a in [hour_to_correct - 2, hour_to_correct + 2]]
                    assert all([col in x.columns for col in cols_to_use])
                    x.loc[idxs, 'hourly_visits_%s.%i' % (date_string, hour_to_correct)] = x.loc[idxs, cols_to_use].mean(axis=1)
        del x['offset_from_gmt']
    x = x.set_index('safegraph_place_id')
    x = x.drop(columns=['visits_by_day'])

    if month_and_year:
        print("%i rows loaded for month and year %s" % (len(x), month_and_year_string))
    else:
        print("%i rows loaded for week %s" % (len(x), week_string))

    return x

def load_google_mobility_data(only_US=True):
    df = pd.read_csv(PATH_TO_GOOGLE_DATA)
    if only_US:
        df = df[df['country_region_code'] == 'US']
    return df

def filter_for_businesses_near_location(df, target_lat, target_lon, distance_in_meters):
    """
    Filter for businesses within distance_in_meters of target_lat, target_lon.
    """
    df = copy.deepcopy(df)
    distance = compute_distance_between_two_lat_lons(lat1=df['latitude'].values,
                                                    lat2=target_lat,
                                                    lon1=df['longitude'].values,
                                                    lon2=target_lon)
    df['meters_from_event'] = distance
    idxs = distance < distance_in_meters
    print("%2.3f%% of locations (%i locations) are within %2.3f meters" % (
        idxs.mean() * 100,
        int(idxs.sum()),
        distance_in_meters))
    small_df = df.loc[idxs]
    return small_df

def list_datetimes_in_range(min_day, max_day):
    """
    Return a list of datetimes in a range from min_day to max_day, inclusive.
    """
    assert(min_day <= max_day)
    days = []
    while min_day <= max_day:
        days.append(min_day)
        min_day = min_day + datetime.timedelta(days=1)
    return days

def list_hours_in_range(min_hour, max_hour):
    assert(min_hour <= max_hour)
    hours = []
    while min_hour <= max_hour:
        hours.append(min_hour)
        min_hour = min_hour + datetime.timedelta(hours=1)
    return hours

def compute_weekly_averages(days_of_data, y, compute_at_daily_resolution=True):
    """
    Average together data by week. Two ways of doing this depending on whether we have
    compute_at_daily_resolution = True.

    If compute_at_daily_resolution=True, return a timeseries at one-day resolution.
    Take unweighted average of 7-day period surrounding each day.
    Assumes dataset comes from contiguous dates.
    Resulting timeseries will be shorter by 6 days (cutting off 3 at begnning and end) but will have one-
    day resolution.

    If compute_at_daily_resolution=False, return 1/7 as many points: starting at beginning of timeseries,
    compute one point for each week which is the average for the week.
    """
    assert len(y) == len(days_of_data)
    assert all([(days_of_data[i + 1] - days_of_data[i]).days == 1 for i in range(len(days_of_data) - 1)]) # make sure timeseries is contiguous.
    if compute_at_daily_resolution:
        weights = np.ones(7) / 7.
        y = np.convolve(y, weights, mode='valid') # https://gist.github.com/rday/5716218
        assert len(y) == (len(days_of_data) - 6)
        return days_of_data[3:-3], y

    if not len(y) % 7 == 0: # don't want any partially truncated weeks.
        raise Exception("Date range length should be a multiple of 7 for weekly plotting; current length is %i" % len(y))
    min_date = min(days_of_data)
    weeks_since_beginning = np.array([math.floor((a - min_date).days / 7.) for a in days_of_data])

    week_start_dates = []
    weekly_y = []
    for w in sorted(list(set(weeks_since_beginning))):
        week_idxs = weeks_since_beginning == w
        week_start_dates.append(np.array(days_of_data)[week_idxs].min())
        weekly_y.append(np.array(y)[week_idxs].mean())
    return week_start_dates, weekly_y

def compute_timeseries_from_original_df(df, days_of_data, field_name=None, field_values=None,
                                        top_n=5, agg_func=np.sum, divide_each_line_by_mean=False,
                                        average_together_by_week=False):
    """
    This is called as a helper method by several other functions. It was originally part of Serina's
    plotting code.
    Given a dataframe, it computes timeseries grouping together data.
    Returns a dictionary where each key is a group, and values are dictionaries with fields days_of_data,
    y, and n_pois.

    :param - df: Pandas dataframe, the data to compute timeseries for.
    :param - days_of_data: list of datetime.datetime objects, the days to analyze
    :param - field_name: string, the name of the field to breakdown on; if None, then there
                         is no breakdown and the data aggregated over all of df is analyzed
    :param - field_values: list of strings, the name of the field values to show
                           (e.g. ['restaurants', 'banks', 'amusement parks']; if None, then
                           the field values with the most corresponding rows are used
    :param - top_n: int, the top_n field values to analyze, only used if field_values is not specified
    :param - agg_func: function, the data aggregation function
    :param - divide_each_line_by_mean: divide timeseries line by its mean to put things on the same scale.
    :param - average_together_by_week: compute weekly averages.
    """
    day_cols = [a.strftime('%Y.%-m.%-d') for a in days_of_data]
    assert(all([col in df.columns for col in day_cols]))
    timeseries_to_return = {}

    if field_name is None:  # no breakdown. Add helper column here to avoid duplicating code.
        df['all_data'] = True
        field_name = 'all_data'
        field_values = [True]

    # Group dataframe and loop over groups.
    val2idx = df.groupby(field_name).indices
    if field_values is None:
        vals = sorted(val2idx.keys(), key=lambda x:len(val2idx[x]), reverse=True)
        field_values = vals[:top_n]  # top n most frequent values within field
    for i, val in enumerate(field_values):
        timeseries_to_return['%s=%s' % (field_name, val)] = {}
        if not (val in val2idx):
            raise Exception(f"Failed for {val}; {field_values}")
        idx = val2idx[val]
        rows = df.iloc[idx]
        rows = rows[day_cols].dropna().to_numpy()
        if len(rows) != len(idx):
            print("Warning: dropped %i/%i rows for %s because they contained NAs" % (len(idx) - len(rows), len(idx), val))
        all_visits = agg_func(rows, axis=0)
        if divide_each_line_by_mean:
            all_visits = all_visits/all_visits.mean()
        if average_together_by_week:
            days_to_plot, all_visits = compute_weekly_averages(days_of_data, all_visits) # note new variable name, days_to_plot - avoid overwriting original data.
        else:
            days_to_plot = days_of_data
        timeseries_to_return['%s=%s' % (field_name, val)]['days_of_data'] = days_to_plot
        timeseries_to_return['%s=%s' % (field_name, val)]['y'] = all_visits
        timeseries_to_return['%s=%s' % (field_name, val)]['n_pois'] = len(rows)
    return timeseries_to_return

def plot_visit_count_breakdown_by_field(df, days_of_data, title_string, field_name=None,
                                        field_values=None, top_n=5, figsize=(6,4), agg_func=np.sum,
                                        divide_each_line_by_mean=False,
                                        average_together_by_week=False,
                                        x_interval='week',
                                        ylim=None,
                                        intervention_dates_to_plot=None,
                                        return_fig_and_ax_as_well=False,
                                        fig=None,
                                        ax=None):
    """
    Plots the visit counts per day, broken down by the field_name if provided.
    For all argument descriptions, see compute_timeseries_from_original_df.
    :param - df: Pandas dataframe, the data to visualize
    :param - ylim: set the limits on the yaxis.
    :param - intervention_dates_to_plot: plot intervention dates.
    """

    assert(x_interval in {'month', 'week', 'day'})
    timeseries_to_plot = compute_timeseries_from_original_df(df=df,
        days_of_data=days_of_data,
        field_name=field_name,
        field_values=field_values,
        top_n=top_n,
        agg_func=agg_func,
        divide_each_line_by_mean=divide_each_line_by_mean,
        average_together_by_week=average_together_by_week)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if field_name is None:  # no breakdown
        ax.plot_date(timeseries_to_plot['all_data=True']['days_of_data'], timeseries_to_plot['all_data=True']['y'], fmt='-', linestyle='-', color='gray', linewidth=3)
    else:
        field_values = timeseries_to_plot.keys()
        for val in field_values:
            ax.plot_date(timeseries_to_plot[val]['days_of_data'],
                timeseries_to_plot[val]['y'],
                fmt='-',
                linestyle='-',
                label='%s (n=%i)' % (val, timeseries_to_plot[val]['n_pois']),
                linewidth=3)
        plt.legend(loc='best')

    if x_interval == 'month':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        x_label = 'Month'
    elif x_interval == 'week':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        x_label = 'Week (marked on Mondays)'
    else:
        ax.xaxis.set_major_location(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        x_label = 'Day'

    fig.autofmt_xdate()
    plt.grid(alpha=.2)
    plt.title(title_string)

    plt.xlabel(x_label)
    plt.ylabel("Number of visits (%s; divide by mean=%s)" % (agg_func.__name__, divide_each_line_by_mean))
    if ylim is not None:
        plt.ylim(ylim)
    if intervention_dates_to_plot is not None:
        for k in intervention_dates_to_plot:
            ax.plot_date([k, k], ax.get_ylim(), fmt='-', color='black', linestyle='--')

    if not return_fig_and_ax_as_well:
        plt.show()
        return timeseries_to_plot
    else:
        return timeseries_to_plot, fig, ax

def compute_coronavirus_effect_with_prophet(dates, y, coronavirus_start_date, yearly_seasonality=True,
                                            make_plot=False, do_full_mcmc=False):
    """
    Given a list of dates, and a timeseries y, and a coronavirus start date, fits a model estimating an
    effect of coronavirus [as an indicator variable].
    """
    assert(coronavirus_start_date >= dates[0] and coronavirus_start_date <= dates[-1])
    assert(len(dates) == len(y)), 'length of dates = %d, length of y = %d' % (len(dates), len(y))
    t0 = time.time()
    df = pd.DataFrame({'ds':dates, 'y':y})
    df['post_coronavirus'] = df['ds'] >= coronavirus_start_date

    if do_full_mcmc:
        prophet_model = Prophet(yearly_seasonality=yearly_seasonality, interval_width=0.95,
                                mcmc_samples=500, seasonality_mode='multiplicative')
    else:
        prophet_model = Prophet(yearly_seasonality=yearly_seasonality, interval_width=0.95,
                                seasonality_mode='multiplicative')
    prophet_model.add_regressor('post_coronavirus', mode='multiplicative')
    prophet_model.fit(df)
    prediction = prophet_model.predict(df)
    y = df['y'].values
    yhat = prediction['yhat'].values
    r2 = pearsonr(y, yhat)[0]**2
    mean_percent_error = 100 * np.mean((y - yhat) / y)
    mean_percent_absolute_error = 100 * np.mean(np.abs((y - yhat) / y))

    coronavirus_effect = prediction['post_coronavirus'].iloc[-1]  # because post_coronavirus is an indicator variable, this column only takes two values.
    assert len(set(prediction['post_coronavirus'])) == 2
    assert prediction['post_coronavirus'].iloc[0] == 0
    if not do_full_mcmc:
        coronavirus_lower_ci = None
        coronavirus_upper_ci = None
    else:
        coronavirus_lower_ci = prediction['post_coronavirus_lower'].iloc[-1]
        coronavirus_upper_ci = prediction['post_coronavirus_upper'].iloc[-1]
    if make_plot:
        fig1 = prophet_model.plot(prediction)
        fig2 = prophet_model.plot_components(prediction)
    print("Seconds to fit model: %2.3f" % (time.time() - t0))
    metrics = {'r2':r2,
               'mean_percent_error':mean_percent_error,
               'mean_percent_absolute_error':mean_percent_absolute_error,
               'virus_beta':coronavirus_effect,
               'virus_lower_ci':coronavirus_lower_ci,
               'virus_upper_ci':coronavirus_upper_ci}
    print("Results: virus beta %2.3f; r2 %2.3f" % (metrics['virus_beta'], metrics['r2']))

    return metrics

def run_diff_in_diff(df, days_of_data, title_string, field_name, field_values, intervention_date,
                     which_timeseries_got_interventions, average_together_by_week=False, additional_changepoints=None, title=None, fig=None, ax=None):
    """
    Compute diff-in-diff estimate of the causal effect of an intervention. Does weekly smoothing.
    df: dataframe to use for estimtaes.
    days_of_data: dates to examine.
    title_string: title for the plot, required by plot_visit_count_breakdown_by_field.
    field_name: the field to group by.
    field_values: the values of the field of interest.
    intervention_date: when the intervention occurs.
    which_timeseries_got_interventions: list of which timeseries received interventions.
    """

    # Make timeseries plot.
    # useful for sanity-checking parallel trends assumption.
    # Note this automatically smooths the data because average_together_by_week=True.
    assert len(field_values) == 2
    if additional_changepoints is None:
        additional_changepoints = []
    individual_timeseries, fig, ax = plot_visit_count_breakdown_by_field(df,
                                                                days_of_data,
                                                                title_string=title_string,
                                                                field_name=field_name,
                                                                field_values=field_values,
                                                                average_together_by_week=average_together_by_week,
                                                                divide_each_line_by_mean=True,
                                                                intervention_dates_to_plot=[intervention_date],
                                                                figsize=(8, 4),
                                                                return_fig_and_ax_as_well=True,
                                                                fig=fig,
                                                                ax=ax)

    regression_df = []
    for k in individual_timeseries:
        regression_df.append(pd.DataFrame({'timeseries_label':k,
                                'y':individual_timeseries[k]['y'],
                                'date':individual_timeseries[k]['days_of_data']}))
    regression_df = pd.concat(regression_df)
    min_date = regression_df['date'].min()
    regression_df['t'] = regression_df['date'].map(lambda x:(x - min_date).days) # for time trend.

    changepoints_covars = []
    for i, changepoint_date in enumerate(additional_changepoints):
        regression_df['changepoint_%i' % i] = regression_df['date'].map(lambda x:max((x - changepoint_date).days, 0))
        changepoints_covars.append('changepoint_%i' % i)
    if len(changepoints_covars) == 0:
        changepoint_string = ''
    else:
        changepoint_string = '+' + '+'.join(changepoints_covars)


    regression_df['weekday'] = regression_df['date'].map(lambda x:x.strftime('%A'))


    regression_df['got_intervention'] = regression_df['timeseries_label'].map(lambda x:x in which_timeseries_got_interventions)
    regression_df['post_intervention'] = ((regression_df['date'] >= intervention_date) & regression_df['got_intervention'])# this is coefficient of interest.

    regression_df['post_intervention_but_did_not_receive_intervention'] = ((regression_df['date'] >= intervention_date) & (~regression_df['got_intervention']))

    assert regression_df['post_intervention'].sum() > 0
    if average_together_by_week:
        model = sm.OLS.from_formula('y ~ C(timeseries_label) + t + post_intervention %s' % changepoint_string, data=regression_df).fit()
    else:
        model = sm.OLS.from_formula('y ~ C(timeseries_label) + t + post_intervention + C(weekday) %s' %  changepoint_string, data=regression_df).fit()


    for got_intervention in [True, False]:
        model_predicted_vals = model.predict(regression_df.loc[regression_df['got_intervention'] == got_intervention])
        ax.plot_date(regression_df.loc[regression_df['got_intervention'] == got_intervention, 'date'], model_predicted_vals,
                    fmt='-', label='regression model, intervention=%s' % got_intervention, alpha=.8, linestyle='--', color='black' if got_intervention else 'grey')
    plt.legend()

    idf, cdf = regression_df[regression_df['got_intervention']], regression_df[~regression_df['got_intervention']]
    ddf = pd.merge(idf, cdf, on='date')
    ddf = ddf[(ddf['date'] < intervention_date)]
    dif = ddf['y_x'] - ddf['y_y']
    level = dif.mean() # this should ideally be zero, but if its not its ok as long as the trend variance is low
    trend_var = dif.std() # should be as close to zero as possible
    post_intervention_mean_got_intervention = (regression_df.loc[regression_df['post_intervention'] == True, 'y'].mean())
    post_intervention_mean_no_intervention = (regression_df.loc[regression_df['post_intervention_but_did_not_receive_intervention'] == True, 'y'].mean())
    plt.title((title_string + '\nRegression estimated effect: %2.3f; post intervention difference in means %2.3f\nrelative effect %2.0f%% (regression), %2.0f%% (simple means)') % (
        model.params['post_intervention[T.True]'],
        post_intervention_mean_got_intervention - post_intervention_mean_no_intervention - level,
        100 * np.abs(model.params['post_intervention[T.True]']/post_intervention_mean_no_intervention),
        100 * np.abs((post_intervention_mean_got_intervention - post_intervention_mean_no_intervention)/post_intervention_mean_no_intervention)))
    if title is not None:
        plt.savefig(title)
        plt.close()
    else:
        plt.show()

    return [model.params['post_intervention[T.True]'], post_intervention_mean_got_intervention - post_intervention_mean_no_intervention - level, 100 * (model.params['post_intervention[T.True]']/post_intervention_mean_no_intervention), 100 * ((post_intervention_mean_got_intervention - post_intervention_mean_no_intervention)/post_intervention_mean_no_intervention), level, trend_var]



def compute_timeseries_relative_to_last_year(df, field_name=None, field_values=None, agg_func=np.sum,
                                             top_n=None, make_plot=False, min_poi_cutoff=10,
                                             first_day_to_match=40):
    """
    Normalize this year's timeseries relative to 2019's. First apply weekly smoothing; then take the
    ratio of 2019/2020 timeseries (after multiplying 2019 by a normalization factor
    to make it comparable to 2020).
    """
    assert(first_day_to_match <= 365)  # must be within the year
    max_2020_date = datetime.datetime(2020, 3, 21)  # where to stop analysis in 2020.
    timeseries_to_analyze = compute_timeseries_from_original_df(df=df,
        days_of_data=list_datetimes_in_range(datetime.datetime(2019, 1, 1), max_2020_date),
        field_name=field_name,
        field_values=field_values,
        top_n=top_n,
        agg_func=agg_func,
        divide_each_line_by_mean=True,
        average_together_by_week=True)
    results = {}

    for k in timeseries_to_analyze:
        if timeseries_to_analyze[k]['n_pois'] < min_poi_cutoff:
            continue
        results[k] = {}
        date_arr = np.array(timeseries_to_analyze[k]['days_of_data'])
        y_arr = np.array(timeseries_to_analyze[k]['y'])
        idxs_2019 = np.array([a.year == 2019 for a in date_arr])
        idxs_2020 = np.array([a.year == 2020 and not (a.month == 2 and a.day == 29) for a in date_arr]) # leave out leap year because annoying.
        y_2019 = y_arr[idxs_2019]
        y_2020 = y_arr[idxs_2020]
        y_2019 = y_2019[:len(y_2020)]
        assert len(y_2019) == len(y_2020)

        first_day_date = datetime.datetime(2020, 1, 1) + datetime.timedelta(days=(first_day_to_match-1))
        print('Setting timeseries to be equal (on average) for the week of %s' % first_day_date.strftime('%-m/%-d/%Y'))
        y_2020_match_week = y_2020[first_day_to_match:first_day_to_match+7].mean()
        y_2019_match_week = y_2019[first_day_to_match:first_day_to_match+7].mean()
        normalization_factor = y_2020_match_week / y_2019_match_week  # normalize by setting a week early in the year to be the same on average; default is 40th day (Feb 9)
        normalized_y_2019 = y_2019 * normalization_factor
        ratio = y_2020 / normalized_y_2019
        results[k]['ratio'] = ratio
        results[k]['dates'] = date_arr[idxs_2020]
        results[k]['n_pois'] = timeseries_to_analyze[k]['n_pois']

        if make_plot:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            # first plot: the numerator and denominator of the ratio.
            ax[0].plot_date(date_arr[idxs_2020], normalized_y_2019, fmt='-', label='2019', color='grey')
            ax[0].plot_date(date_arr[idxs_2020], y_2020, fmt='-', label='2020', color='black')
            # second plot: ratio.
            ax[1].plot_date(date_arr[idxs_2020], ratio, fmt='-', label='2020/2019 ratio', color='red', linewidth=3)
            ax[1].plot_date([date_arr[idxs_2020].min(), date_arr[idxs_2020].max()], [1, 1], linestyle='--', fmt='-', color='grey')
            for ax_idx in range(2):
                ax[ax_idx].xaxis.set_major_locator(mdates.MonthLocator())
                ax[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax[ax_idx].legend()
            fig.autofmt_xdate()
            plt.title('%s (n=%i)' % (k, timeseries_to_analyze[k]['n_pois']))
            plt.ylim([.5, 1.5])
            plt.show()
    print("Had sufficient data to compute ratios for %i/%i timeseries (n POI cutoff: %i)" % (len(results), len(timeseries_to_analyze), min_poi_cutoff))
    return results

def fit_prophet_model_by_field(df,
                               days_of_data,
                               coronavirus_start_date,
                               field_name=None,
                               field_values=None,
                               agg_func=np.sum,
                               top_n=None,
                               do_full_mcmc=False):
    """
    By design, this has a call signature quite similar to plot_visit_count_breakdown_by_field.
    Similarly, it calls compute_timeseries_from_original_df.
    """

    timeseries_to_analyze = compute_timeseries_from_original_df(df=df,
        days_of_data=days_of_data,
        field_name=field_name,
        field_values=field_values,
        top_n=top_n,
        agg_func=agg_func,
        divide_each_line_by_mean=False,
        average_together_by_week=False)

    results_df = []
    field_values = sorted(timeseries_to_analyze.keys(), key=lambda x:timeseries_to_analyze[x]['n_pois'])[::-1]
    for i, val in enumerate(field_values):
        print("Fitting prophet model for %s with %i rows" % (val, timeseries_to_analyze[val]['n_pois']))
        results = compute_coronavirus_effect_with_prophet(dates=timeseries_to_analyze[val]['days_of_data'],
            y=timeseries_to_analyze[val]['y'],
            coronavirus_start_date=coronavirus_start_date,
            make_plot=len(field_values) <= 5, # only make plot if there aren't a billion fields
            do_full_mcmc=do_full_mcmc)
        results['field_name'] = field_name
        results['field_value'] = val
        results['n_pois'] = timeseries_to_analyze[val]['n_pois']
        results_df.append(results)
    results_df = pd.DataFrame(results_df)

    return results_df

def load_acs_data(csv_filename):
    """
    Loads a csv file from the ACS directory, e.g. ACS_17_5YR_S1901_with_ann.csv,
    which has annotations for income levels.
    """
    df = pd.read_csv(os.path.join(ACS_DIR, csv_filename), header=1)
    # remove empty columns
    for col in df.columns:
        sample = df.head(100)[col].unique()
        if len(sample) == 1 and sample[0] == '(X)':
            del df[col]
    return df

def get_acs_zipcode_to_median(df):
    """
    Parses the zipcode and median income columns of the ACS dataframe, and ignores
    the rows where the true median is not known.
    """
    zipcode_col = 'Geography'
    median_col = 'Households; Estimate; Median income (dollars)'
    assert(zipcode_col in df.columns and median_col in df.columns)
    zipcode2median = {}
    for zipcode, median in zip(df[zipcode_col], df[median_col]):
        median = re.sub('\D', '', median)  # strip non-digit characters, e.g. '2,500-'
        if len(median) > 0:  # sometimes there is only '-'
            median = int(median)
            zipcode = int(zipcode.split()[1])
            zipcode2median[zipcode] = median
    print('Kept %d tuples out of %d' % (len(zipcode2median), len(df)))
    return zipcode2median

def get_top_and_bottom_quartile_POIs(df, zipcode2median):
    """
    Matches the zipcodes in the POI dataframe to median income and identifies
    the zipcodes in the top and bottom income quartiles of this population.
    Returns the POIs in these two sets of zipcodes.
    """
    zipcodes = set(df.postal_code.unique()).intersection(set(zipcode2median.keys()))
    print('Number of unique zipcodes found:', len(zipcodes))
    medians = sorted([zipcode2median[z] for z in zipcodes])

    cutoff = medians[int(len(medians) * .25)]
    print('25th percentile:', cutoff)
    poor_zipcodes = set([z for z in zipcodes if zipcode2median[z] < cutoff])
    print('Num zipcodes in bottom quartile:', len(poor_zipcodes))
    poor_pois = df[df['postal_code'].isin(poor_zipcodes)]
    print('Num POIs in bottom quartile zipcodes:', len(poor_pois))

    cutoff = medians[int(len(medians) * .75)]
    print('75th percentile:', cutoff)
    rich_zipcodes = [z for z in zipcodes if zipcode2median[z] > cutoff]
    print('Num zipcodes in top quartile:', len(rich_zipcodes))
    rich_pois = df[df['postal_code'].isin(rich_zipcodes)]
    print('Num POIs in top quartile zipcodes:', len(rich_pois))
    return poor_pois, rich_pois

def compare_frequencies_broken_down_by_field(poor_pois, rich_pois, field_name,
                                             top_n=10, smoothing=1):
    """
    Compares the frequencies by field name between POIs from poor zipcodes and
    POIs from rich zipcodes, e.g. the frequency of different brands. The comparison
    score is computed as the frequency within the rich POIs divided by the sum of
    the rich and poor frequencies, so a score closer to 1 indicates a lean towards
    rich areas and a score closer to 0 indicates a lean towards poor areas.
    """
    orig_count = len(poor_pois) + len(rich_pois)
    poor_pois = poor_pois.dropna(subset=[field_name])
    rich_pois = rich_pois.dropna(subset=[field_name])
    new_count = len(poor_pois) + len(rich_pois)
    print('Orig count = %d, new count = %d -> dropped %d POIs with NaN in field' % (orig_count, new_count, orig_count - new_count))

    rich_counts = dict(rich_pois.groupby(field_name).size())
    poor_counts = dict(poor_pois.groupby(field_name).size())
    all_values = set(rich_counts.keys()).union(set(poor_counts.keys()))
    print('%d unique values in %s' % (len(all_values), field_name))
    val2ratio = {}
    for val in all_values:
        rich_count = rich_counts[val] + smoothing if val in rich_counts else smoothing
        rich_prop = rich_count / (len(rich_pois) + (smoothing*len(all_values)))
        poor_count = poor_counts[val] + smoothing if val in poor_counts else smoothing
        poor_prop = poor_count / (len(poor_pois) + (smoothing*len(all_values)))
        ratio = rich_prop / (poor_prop + rich_prop)
        val2ratio[val] = ratio

    vals = sorted(all_values, key=lambda x:val2ratio[x])
    print('\nTop %d most poor-leaning' % top_n)
    for i in range(top_n):
        val = vals[i]
        print(i+1, val, round(val2ratio[val], 3))
    print('\nTop %d most rich-leaning' % top_n)
    for i in range(1, top_n+1):
        val = vals[-i]
        print(i, val, round(val2ratio[val], 3))

def load_election_results_data():
    """
    Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
    MIT Election Data and Science Lab, 2018, "County Presidential Election Returns 2000-2016",
    """
    print("Loading 2016 election results")
    d = pd.read_csv('/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/countypres_2000-2016.csv')
    d = d.loc[(d['year'] == 2016) & (d['party'] == 'republican')]
    d['republican_2016_vote_share'] = d['candidatevotes'] / d['totalvotes']
    d = d[['state_po', 'FIPS', 'republican_2016_vote_share']]
    d.columns = ['state', 'county_fips_code', 'republican_2016_vote_share']
    d['county_fips_code'] = d['county_fips_code'].map(lambda x:x % 1000) # for consistency with safegraph data mapping.
    print("Prior to dropping any county returns, %i rows" % len(d))
    d = d.dropna()
    print("After dropping, %i rows" % len(d))
    assert d[['state', 'county_fips_code']].duplicated().sum() == 0
    return d

def normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints(old_dict):
    """
    Self-explanatory; used by aggregate_visitor_home_cbgs_over_months.
    """
    new_dict = {}
    value_sum = 1.*sum(old_dict.values())
    if len(old_dict) > 0:
        assert value_sum > 0
    for k in old_dict:
        new_dict[int(k)] = old_dict[k] / value_sum
    return new_dict

def cast_keys_to_ints(old_dict):
    new_dict = {}
    for k in old_dict:
        new_dict[int(k)] = old_dict[k]
    return new_dict

def aggregate_visitor_home_cbgs_over_months(d, cutoff_year=2019, population_df=None,
                                            in_place=True, periods_to_include=None):
    """
    Aggregate visitor_home_cbgs across months and produce a normalized aggregate field.

    Usage: d = aggregate_visitor_home_cbgs_over_months(d).
    cutoff = the earliest time (could be year or year.month) to aggregate data from
    population_df = the DataFrame loaded by load_dataframe_to_correct_for_population_size

    Currently only using a couple months because aggregation is slow.
    """
    t0 = time.time()
    if periods_to_include is not None:
        cols = ['%s.visitor_home_cbgs' % period for period in periods_to_include]
    else:
        # Not using CBG data from weekly files for now because of concerns that it's inconsistently
        # processed - they change how they do the privacy filtering.
        weekly_cols_to_exclude = ['%s.visitor_home_cbgs' % a for a in ALL_WEEKLY_STRINGS]
        cols = [a for a in d.columns if (a.endswith('.visitor_home_cbgs') and (a >= str(cutoff_year)) and (a not in weekly_cols_to_exclude))]
    print('Aggregating data from: %s' % cols)

    if not in_place:
        d = d[cols].copy()  # don't modify the df in place, return a copy with the new columns

    # Helper variables to use if visitor_home_cbgs counts need adjusting
    adjusted_cols = []
    if population_df is not None:
        int_cbgs = [int(cbg) for cbg in population_df.census_block_group]

    for k in cols:
        if type(d.iloc[0][k]) != Counter:
            print('Filling %s with Counter objects' % k)
            d[k] = d[k].fillna('{}').map(lambda x:Counter(cast_keys_to_ints(json.loads(x))))  # map strings to counters.
        if population_df is not None:
            sub_t0 = time.time()
            new_col = '%s_adjusted' % k
            if new_col not in d.columns:
                total_population = population_df.total_cbg_population.to_numpy()
                time_period = k.strip('.visitor_home_cbgs')
                population_col = 'number_devices_residing_%s' % time_period
                assert(population_col in population_df.columns)
                num_devices = population_df[population_col].to_numpy()
                cbg_coverage = num_devices / total_population
                median_coverage = np.median(cbg_coverage)
                cbg_coverage = dict(zip(int_cbgs, cbg_coverage))
                # want to make sure we aren't missing data for too many CBGs, so a small hack - have
                # adjust_home_cbg_counts_for_coverage return two arguments, where the second argument
                # tells us if we had to clip or fill in the missing coverage number.
                d[new_col] = d[k].map(lambda x:adjust_home_cbg_counts_for_coverage(x, cbg_coverage, median_coverage=median_coverage))
                print('Finished adjusting home CBG counts for %s [time=%.3fs] had to fill in or clip coverage for %2.6f%% of rows; in those cases used median coverage %2.3f' %
                      (time_period, time.time() - sub_t0, 100 * d[new_col].map(lambda x:x[1]).mean(), median_coverage))
                d[new_col] = d[new_col].map(lambda x:x[0]) # remove the second argument of adjust_home_cbg_counts_for_coverage, we don't need it anymore.
            else:
                print('Adjusted home CBG counts for %s are already computed' % time_period)
            adjusted_cols.append(new_col)

    # add counters together across months.
    d['aggregated_visitor_home_cbgs'] = d[cols].aggregate(func=sum, axis=1)
    # normalize each counter so its values sum to 1.
    d['aggregated_visitor_home_cbgs'] = d['aggregated_visitor_home_cbgs'].map(normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints)

    if len(adjusted_cols) > 0:
        d['aggregated_cbg_population_adjusted_visitor_home_cbgs'] = d[adjusted_cols].aggregate(func=sum, axis=1)
        d['aggregated_cbg_population_adjusted_visitor_home_cbgs'] = d['aggregated_cbg_population_adjusted_visitor_home_cbgs'].map(normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints)
        d = d.drop(columns=adjusted_cols)

    print("Aggregating CBG visitors over %i time periods took %2.3f seconds" % (len(cols), time.time() - t0))
    print("Fraction %2.3f of POIs have CBG visitor data" % (d['aggregated_visitor_home_cbgs'].map(lambda x:len(x) != 0).mean()))
    return d

def adjust_home_cbg_counts_for_coverage(cbg_counter, cbg_coverage, median_coverage, max_upweighting_factor=100):
    """
    Adjusts the POI-CBG counts from SafeGraph to estimate the true count, based on the
    coverage that SafeGraph has for this CBG.
    cbg_counter: a Counter object mapping CBG to the original count
    cbg_coverage: a dictionary where keys are CBGs and each data point represents SafeGraph's coverage.
    This should be between 0 and 1 for the vast majority of cases, although for some weird CBGs it may not be.

    Returns the adjusted dictionary and a Bool flag had_to_guess_coverage_value which tells us whether we had to adjust the coverage value.
    """
    had_to_guess_coverage_value = False
    if len(cbg_counter) == 0:
        return cbg_counter, had_to_guess_coverage_value
    new_counter = Counter()
    for cbg in cbg_counter:
        if cbg not in cbg_coverage:
            upweighting_factor = 1 / median_coverage
            had_to_guess_coverage_value = True
        else:
            upweighting_factor = 1 / cbg_coverage[cbg]  # need to invert coverage
            if upweighting_factor > max_upweighting_factor:
                upweighting_factor = 1 / median_coverage
                had_to_guess_coverage_value = True
        new_counter[cbg] = cbg_counter[cbg] * upweighting_factor
    return new_counter, had_to_guess_coverage_value

def compute_weighted_mean_of_cbg_visitors(cbg_visitor_fracs, cbg_values):
    """
    Given a dictionary cbg_visitor_fracs which gives the fraction of people from a CBG which visit a POI
    and a dictionary cbg_values which maps CBGs to values, compute the weighted mean for the POI.
    """
    if len(cbg_visitor_fracs) == 0:
        return None
    else:
        numerator = 0.
        denominator = 0.
        for cbg in cbg_visitor_fracs:
            if cbg not in cbg_values:
                continue
            numerator += cbg_visitor_fracs[cbg] * cbg_values[cbg]
            denominator += cbg_visitor_fracs[cbg]
        if denominator == 0:
            return None
        return numerator/denominator

def sort_by_decrease_relative_to_last_year(timeseries, min_date, max_date, min_poi_cutoff=10):
    """
    Given the timeseries returned by compute_timeseries_relative_to_last_year,
    computes the decrease between min_date and max_date for each timeseries relative to last year .
    """
    results_df = []

    for k in timeseries:
        if timeseries[k]['n_pois'] < min_poi_cutoff:
            continue
        date_idxs = (timeseries[k]['dates'] >= min_date) & (timeseries[k]['dates'] <= max_date)
        results_df.append({'group':k,
                           'ratio':timeseries[k]['ratio'][date_idxs].mean(),
                           'n_pois':timeseries[k]['n_pois']})
    results_df = pd.DataFrame(results_df)
    return results_df.sort_values(by='ratio')

def explode(df, lst_cols, fill_value='', preserve_index=False):
    """
    Given a df with some list columns convert into a `melted' format.
    Source: https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726
    """
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def get_county_df(df):
    """Takes a chunk or set of chunks, gets the timeseries dict, converts to melted pandas for easy use
        returns: county df with columns ['FIPS', 'n_pois', 'dates', 'ratio'] and rows for each day for each county
    """
    county_dict = compute_timeseries_relative_to_last_year(df,
                                                           field_name=['poi_lat_lon_state_fips_code',
                                                                       'poi_lat_lon_county_fips_code'])
    # incredilby hacky way of pulling the fips code out of the string returned by the dict here, it does seem to work?
    # IMPORTANT TODO: some values become nan on merging with gdf; unclear why and must investigate
    county_dict = {np.sum(np.array(ast.literal_eval(k.split('=')[1])).astype(float) * np.array([1000, 1])): county_dict[k]
                   for k in county_dict}
    df = pd.DataFrame(county_dict).transpose()
    df['FIPS'] = df.index
    return explode(df, ['dates', 'ratio'])

def get_gdf(county_df):
    """Merges the county df returned by the get_county_df function with the US counties shapefile to allow plotting with geopandas -- currently exclude alaska and hawaii because they map badly (we should fix this)"""
    gdf = gpd.read_file('/dfs/scratch1/safegraph_homes/county_data/UScounties/UScounties.shp')
    gdf = gdf[(gdf['STATE_NAME'] != 'Alaska') & (gdf['STATE_NAME'] != 'Hawaii')]
    gdf['FIPS'] = gdf['FIPS'].astype(int)
    county_df['FIPS'] = county_df['FIPS'].astype(int)
    # IMPORTANT TODO: some values become nan on merging with gdf; unclear why and must investigate; there are counties in the county df not corresponding to counties in the shape file... likley this is an Alaska/Hawaii problem... need to check
    return gdf.merge(county_df, how='right', on = "FIPS")

def get_gdf_covs(gdf):
    """Merges the gdf returned by get_gdf with the county covariates"""
    county_data = pd.read_csv('/dfs/scratch1/safegraph_homes/county_data/final-2016-county-data.csv')
    county_data['FIPS'] = county_data['fips'].astype(int)
    return gdf.merge(county_data, how='left', on='FIPS')

def compute_areas_of_polygons_in_square_feet(d):
    # first have to load as shapes.
    print("Computing areas of each polygon in square feet")
    loaded_polygons = d['polygon_wkt'].map(wkt.loads)
    gdf = gpd.GeoDataFrame(loaded_polygons,
                           geometry='polygon_wkt',
                           crs={'init': 'epsg:4326'}) # CRS comes from SG documentation, "Spatial Reference used: EPSG:4326"

    # now compute areas.

    # https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas
    # See comments about using cea projection.
    gdf = gdf.to_crs({'proj':'cea'})
    area_in_square_meters = gdf['polygon_wkt'].area.values
    area_in_square_feet = 10.7639 * area_in_square_meters
    d['area_in_square_feet'] = area_in_square_feet
    print(d['area_in_square_feet'].describe())
    return d

def load_dataframe_for_individual_msa(MSA_name, nrows=None):
    """
    This loads all the POI info for a single MSA.
    """
    t0 = time.time()
    filename = os.path.join(STRATIFIED_BY_AREA_DIR, '%s.csv' % MSA_name)
    d = pd.read_csv(filename, nrows=nrows)
    for k in (['aggregated_cbg_population_adjusted_visitor_home_cbgs', 'aggregated_visitor_home_cbgs']):
        d[k] = d[k].map(lambda x:cast_keys_to_ints(json.loads(x)))
    for k in ['%s.visitor_home_cbgs' % a for a in ALL_WEEKLY_STRINGS]:
        if k in d.columns:
            d[k] = d[k].fillna('{}')
            d[k] = d[k].map(lambda x:cast_keys_to_ints(json.loads(x)))
        else:
            print('Warning: this dataframe is missing column %s' % k)
    print("Loaded %i rows for %s in %2.3f seconds" % (len(d), MSA_name, time.time() - t0))
    return d

def load_dataframe_to_correct_for_population_size(just_load_census_data=False):
    """
    Load in a dataframe with rows for the 2018 ACS Census population code in each CBG
    and the SafeGraph population count in each CBG (from home-panel-summary.csv).

    The correlation is not actually that good. Is this just noise [in both SafeGraph and 1-year ACS?]

    Definition of
    num_devices_residing: Number of distinct devices observed with a primary nighttime location in the specified census block group.
    """
    acs_data = pd.read_csv(PATH_TO_ACS_1YR_DATA,
                          encoding='cp1252',
                       usecols=['STATEA', 'COUNTYA', 'TRACTA', 'BLKGRPA','AJWBE001'],
                       dtype={'STATEA':str,
                              'COUNTYA':str,
                              'BLKGRPA':str,
                             'TRACTA':str})
    # https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
    # STATE+COUNTY+TRACT+BLOCK GROUP
    assert (acs_data['STATEA'].map(len) == 2).all()
    assert (acs_data['COUNTYA'].map(len) == 3).all()
    assert (acs_data['TRACTA'].map(len) == 6).all()
    assert (acs_data['BLKGRPA'].map(len) == 1).all()
    acs_data['census_block_group'] = (acs_data['STATEA'] +
                                    acs_data['COUNTYA'] +
                                    acs_data['TRACTA'] +
                                    acs_data['BLKGRPA'])
    acs_data['census_block_group'] = acs_data['census_block_group'].astype(int)
    assert len(set(acs_data['census_block_group'])) == len(acs_data)
    acs_data['county_code'] = (acs_data['STATEA'] + acs_data['COUNTYA']).astype(int)
    acs_data = acs_data[['census_block_group', 'AJWBE001', 'STATEA', 'county_code']]
    acs_data = acs_data.rename(mapper={'AJWBE001':'total_cbg_population',
                                       'STATEA':'state_code'}, axis=1)
    print("%i rows of 2018 1-year ACS data read" % len(acs_data))
    if just_load_census_data:
        return acs_data
    combined_data = acs_data


    # now read in safegraph data to use as normalizer. Months and years first.
    all_filenames = []
    all_date_strings = []
    for month, year in [(1, 2017),(2, 2017),(3, 2017),(4, 2017),(5, 2017),(6, 2017),(7, 2017),(8, 2017),(9, 2017),(10, 2017),(11, 2017),(12, 2017),
             (1, 2018),(2, 2018),(3, 2018),(4, 2018),(5, 2018),(6, 2018),(7, 2018),(8, 2018),(9, 2018),(10, 2018),(11, 2018),(12, 2018),
             (1, 2019),(2, 2019),(3, 2019),(4, 2019),(5, 2019),(6, 2019),(7, 2019),(8, 2019),(9, 2019),(10, 2019),(11, 2019),(12, 2019),
             (1, 2020),(2, 2020)]:
        if (year == 2019 and month == 12) or (year == 2020 and month in [1, 2]):
            upload_date_string = '2020-03-16'  # we uploaded files in two groups; load them in the same way.
        else:
            upload_date_string = '2019-12-12'
        month_and_year_string = '%i_%02d-%s' % (year, month, upload_date_string)
        filename = os.path.join(UNZIPPED_DATA_DIR,
                                'SearchofAllRecords-CORE_POI-GEOMETRY-PATTERNS-%s' % month_and_year_string,
                                'home_panel_summary.csv')
        all_filenames.append(filename)
        all_date_strings.append('%i.%i' % (year, month))

    # now weeks
    for date_string in ALL_WEEKLY_STRINGS:
        all_filenames.append(
            '/dfs/scratch1/safegraph_homes/all_aggregate_data/weekly_patterns_data/v1/home_summary_file/%s-home-panel-summary.csv' % date_string)
        all_date_strings.append(date_string)

    cbgs_with_ratio_above_one = np.array([False for a in range(len(acs_data))])

    for filename_idx, filename in enumerate(all_filenames):
        date_string = all_date_strings[filename_idx]
        print("\n*************")
        safegraph_counts = pd.read_csv(filename, dtype={'census_block_group':str})
        print("%s: %i devices read from %i rows" % (
            date_string, safegraph_counts['number_devices_residing'].sum(), len(safegraph_counts)))
        safegraph_counts = safegraph_counts[['census_block_group', 'number_devices_residing']]

        col_name = 'number_devices_residing_%s' % date_string
        safegraph_counts.columns = ['census_block_group', col_name]
        safegraph_counts['census_block_group'] = safegraph_counts['census_block_group'].map(int)
        assert len(safegraph_counts['census_block_group'].dropna()) == len(safegraph_counts)
        print("Number of unique Census blocks: %i; unique blocks %i: WARNING: DROPPING NON-UNIQUE ROWS" %
              (len(safegraph_counts['census_block_group'].drop_duplicates()), len(safegraph_counts)))
        safegraph_counts = safegraph_counts.drop_duplicates(subset=['census_block_group'], keep=False)

        combined_data = pd.merge(combined_data,
                                 safegraph_counts,
                                 how='left',
                                 validate='one_to_one',
                                 on='census_block_group')
        missing_data_idxs = pd.isnull(combined_data[col_name])
        print("Missing data for %i rows; filling with zeros" % missing_data_idxs.sum())
        combined_data.loc[missing_data_idxs, col_name] = 0

        r, p = pearsonr(combined_data['total_cbg_population'], combined_data[col_name])
        combined_data['ratio'] = combined_data[col_name]/combined_data['total_cbg_population']
        cbgs_with_ratio_above_one = cbgs_with_ratio_above_one | (combined_data['ratio'].values > 1)
        combined_data.loc[combined_data['total_cbg_population'] == 0, 'ratio'] = None
        print("Ratio of SafeGraph count to Census count")
        print(combined_data['ratio'].describe(percentiles=[.25, .5, .75, .9, .99, .999]))
        print("Correlation between SafeGraph and Census counts: %2.3f" % (r))
    print("Warning: %i CBGs with a ratio greater than 1 in at least one month" % cbgs_with_ratio_above_one.sum())
    del combined_data['ratio']
    combined_data.index = range(len(combined_data))
    assert len(combined_data.dropna()) == len(combined_data)
    return combined_data

def load_and_reconcile_multiple_acs_data():
    acs_1_year_d = load_dataframe_to_correct_for_population_size(just_load_census_data=True)
    column_rename = {'total_cbg_population':'total_cbg_population_2018_1YR'}
    acs_1_year_d = acs_1_year_d.rename(mapper=column_rename, axis=1)
    acs_1_year_d['state_name'] = acs_1_year_d['state_code'].map(lambda x:FIPS_CODES_FOR_50_STATES_PLUS_DC[str(x)] if str(x) in FIPS_CODES_FOR_50_STATES_PLUS_DC else np.nan)
    acs_5_year_d = pd.read_csv(PATH_TO_ACS_5YR_DATA)
    print('%i rows of 2017 5-year ACS data read' % len(acs_5_year_d))
    acs_5_year_d['census_block_group'] = acs_5_year_d['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
    # rename dynamic attributes to indicate that they are from ACS 2017 5-year
    dynamic_attributes = ['p_black', 'p_white', 'p_asian', 'median_household_income',
                          'block_group_area_in_square_miles', 'people_per_mile']
    column_rename = {attr:'%s_2017_5YR' % attr for attr in dynamic_attributes}
    acs_5_year_d = acs_5_year_d.rename(mapper=column_rename, axis=1)
    # repetitive with 'state_code' and 'county_code' column from acs_1_year_d
    acs_5_year_d = acs_5_year_d.drop(['Unnamed: 0', 'STATEFP', 'COUNTYFP'], axis=1)
    combined_d = acs_1_year_d.join(acs_5_year_d.set_index('census_block_group'), on='census_block_group')
    combined_d['people_per_mile_hybrid'] = combined_d['total_cbg_population_2018_1YR'] / combined_d['block_group_area_in_square_miles_2017_5YR']
    return combined_d

def parse_msa_to_county_mapping():
    msa2county = {}
    with open(MSA_COUNTY_MAPPING, 'r') as f:
        reader = csv.DictReader(f, fieldnames = ['msa_code', 'msa_name', 'county_code', 'county_name'])
        for row in reader:
            msa = row['msa_name']
            if msa is not None:
                msa = msa[:-len(' (Metropolitan Statistical Area)')]
                if msa not in msa2county:
                    msa2county[msa] = [], []
                try:
                    msa2county[msa][0].append(np.int64(row['county_code']))
                    msa2county[msa][1].append(row['county_name'])
                except:
                    continue
    return msa2county

def compute_cbg_day_prop_out(sdm_of_interest, cbgs_of_interest=None):
    '''
    Computes the proportion of people leaving a CBG on each day.
    It returns a new DataFrame, with one row per CBG representing proportions for each day in sdm_of_interest.

    sdm_of_interest: a Social Distancing Metrics dataframe, data for the time period of interest
    cbgs_of_interest: a list, the CBGs for which to compute reweighting; if None, then
                      reweighting is computed for all CBGs in sdm_of_interest

    ---------------------------------------
    Sample usage:

    sdm_sq = helper.load_social_distancing_metrics(status_quo_days)
    days_of_interest = helper.list_datetimes_in_range(datetime.datetime(2020, 3, 1), datetime.datetime(2020, 4, 1))
    sdm_of_interest = helper.load_social_distancing_metrics(days_of_interest)
    reweightings_df = helper.compute_cbg_day_reweighting( sdm_of_interest)

    '''
    # Process SDM of interest dataframe
    orig_len = len(sdm_of_interest)
    interest_num_home_cols = [col for col in sdm_of_interest.columns if col.endswith('completely_home_device_count')]
    interest_device_count_cols = [col for col in sdm_of_interest.columns if col.endswith('device_count') and col not in interest_num_home_cols]
    sdm_of_interest = sdm_of_interest.dropna(subset=interest_device_count_cols + interest_num_home_cols)
    sdm_of_interest.set_index(sdm_of_interest['census_block_group'].values, inplace=True)
    print('Kept %i / %i CBGs with non-NaN SDM for days of interest' % (len(sdm_of_interest), orig_len))

    if cbgs_of_interest is None:
        cbgs_of_interest = sdm_of_interest.census_block_group.unique()
    # Find CBGs in common between SDM dataframe and CBGs of interest
    cbgs_with_data = set(cbgs_of_interest).intersection(sdm_of_interest.index)
    print('Found SDM data for %i / %i CBGs of interest' % (len(cbgs_with_data), len(cbgs_of_interest)))

    # Get proportion of population that goes out during days of interest
    sub_sdm_int = sdm_of_interest[sdm_of_interest['census_block_group'].isin(cbgs_with_data)]
    assert(len(sub_sdm_int) == len(cbgs_with_data))
    sub_sdm_int = sub_sdm_int.sort_values(by='census_block_group')
    int_num_out = sub_sdm_int[interest_device_count_cols].values - sub_sdm_int[interest_num_home_cols].values
    int_prop_out = int_num_out / sub_sdm_int[interest_device_count_cols].values
    int_prop_out = np.clip(int_prop_out, 1e-10, None)  # so that the reweighting is not zero
    N, T = int_prop_out.shape

    dates = [col.strip('_device_count') for col in interest_device_count_cols]
    dates2 = [col.strip('_completely_home_device_count') for col in interest_num_home_cols]
    assert dates == dates2
    sorted_cbgs_with_data = sorted(cbgs_with_data)
    prop_df = pd.DataFrame(int_prop_out, columns=dates)
    prop_df['census_block_group'] = sorted_cbgs_with_data
    # If we could not compute reweighting for a CBG, use median reweighting for that day
    if len(cbgs_with_data) < len(cbgs_of_interest):
        missing_cbgs = set(cbgs_of_interest) - cbgs_with_data
        print('Filling %d CBGs with median props' % len(missing_cbgs))
        median_prop = np.median(int_prop_out, axis=0)
        missing_props = np.broadcast_to(median_prop, (len(missing_cbgs), T))
        missing_props_df = pd.DataFrame(missing_props, columns=dates)
        missing_props_df['census_block_group'] = list(missing_cbgs)
        prop_df = pd.concat((prop_df, missing_props_df))
    return prop_df

def write_out_acs_5_year_data():
    cbg_mapper = dataprocessor.CensusBlockGroups(
        base_directory='/dfs/scratch1/safegraph_homes/old_dfs_scratch0_directory_contents/new_census_data/',
        gdb_files=None)

    geometry_cols = ['STATEFP',
              'COUNTYFP',
              'TRACTCE',
              'Metropolitan/Micropolitan Statistical Area',
              'CBSA Title',
              'State Name',
              'PUMA5CE']
    block_group_cols = ['GEOID',
                              'p_black',
                              'p_white',
                              'p_asian',
                              'median_household_income',
                             'block_group_area_in_square_miles',
                             'people_per_mile']
    for k in geometry_cols:
        cbg_mapper.block_group_d[k] = cbg_mapper.geometry_d[k].values
    df_to_write_out = cbg_mapper.block_group_d[block_group_cols + geometry_cols]
    print("Total rows: %i" % len(df_to_write_out))
    print("Missing data")
    print(pd.isnull(df_to_write_out).mean())
    output_path = '/dfs/scratch1/safegraph_homes/external_datasets_for_aggregate_analysis/2017_five_year_acs_data/2017_five_year_acs_data.csv'
    df_to_write_out.to_csv(output_path)
