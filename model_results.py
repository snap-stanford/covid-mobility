import pandas as pd
import numpy as np
from covid_constants_and_util import *
from model_experiments import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as tick
from collections import Counter
import datetime
from scipy.stats import linregress
import os
import pickle
import math

###################################################################################
# Figure 1: model fit
###################################################################################
def get_best_models_for_all_msas(df, key_to_sort_by, only_single_best=False, 
                                 loss_tolerance=ACCEPTABLE_LOSS_TOLERANCE, verbose=False):
    model_str = 'model' if only_single_best else 'models'
    print('Finding best fit %s based on key=%s, loss tolerance=%2.3f' % (model_str, key_to_sort_by, loss_tolerance))
    best_models = []
    for msa in MSAS:
        msa_df = df[df['MSA_name'] == msa]
        best_loss = np.min(msa_df[key_to_sort_by].values)
        if only_single_best:
            msa_best_models = msa_df[msa_df[key_to_sort_by] == best_loss]
        else:
            msa_best_models = msa_df[msa_df[key_to_sort_by] <= (loss_tolerance * best_loss)]
        if verbose:
            print(msa, len(msa_best_models))
        best_models.append(msa_best_models)
    best_models = pd.concat(best_models)
    return best_models

def compare_best_models(df1, df2, eval_field):
    assert len(df1) == len(df2)
    df1 = df1.sort_values(by='MSA_name')
    df2 = df2.sort_values(by='MSA_name')
    loss1 = df1[eval_field].values
    loss2 = df2[eval_field].values
    winner = loss1 < loss2
    failed_msas = df1.loc[~winner].MSA_name.values  # assume we want df1 to beat df2
    print('first has lower loss than second in %d MSAs; failed on %s' % (np.sum(winner), failed_msas))
    print('avg ratio of first / second = %.3f' % np.mean(loss1 / loss2))
    print('median ratio of first / second = %.3f' % np.median(loss1 / loss2))
    
def get_summary_of_best_models(df, msas, threshold, max_models, loss_key='loss_dict_daily_cases_RMSE'):
    keys = ['home_beta', 'poi_psi', 'p_sick_at_t0']
    columns = ['MSA', 'n_models'] + [f'{key}_min' for key in keys] + [f'{key}_max' for key in keys]
    model_df = pd.DataFrame(columns=columns)
    model_df_idx = 0

    for msa_name in msas:
        subdf = df[(df['MSA_name'] == msa_name)].copy()
        subdf = subdf.sort_values(by=loss_key)
        losses = subdf[loss_key] / subdf.iloc[0][loss_key]
        n_models = min(max_models, np.sum(losses <= threshold))

        print("Spearman correlation matrix for best-fit models")
        print(subdf.iloc[:n_models][keys].corr(method='spearman'))

        model_dict = {}
        model_dict['MSA'] = msa_name
        # Best fit model
        for key in keys:
            model_dict[key] = subdf.iloc[0][key]

        for key in keys:
            model_dict[f'{key}_max'] = -1000000
            model_dict[f'{key}_min'] =  1000000

        # Mins and maxes
        for subdf_idx in range(n_models):
            for key in keys:
                min_key = f'{key}_min'
                max_key = f'{key}_max'
                if model_dict[min_key] > subdf.iloc[subdf_idx][key]:
                    model_dict[min_key] = subdf.iloc[subdf_idx][key]
                if model_dict[max_key] < subdf.iloc[subdf_idx][key]:
                    model_dict[max_key] = subdf.iloc[subdf_idx][key]

        model_dict['n_models'] = n_models
        model_df = model_df.append(model_dict, ignore_index=True)
        print(f'{msa_name:50s}: {n_models:3d}')

    return model_df

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

def plot_best_models_fit_for_msa(df, msa_name, ax, key_to_sort_by, train_test_partition,
                                 plotting_kwargs, old_directory=False):

    subdf = df[(df['MSA_name'] == msa_name)].copy()
    subdf = subdf.sort_values(by=key_to_sort_by)
    losses = subdf[key_to_sort_by] / subdf.iloc[0][key_to_sort_by]
    num_models_to_aggregate = np.sum(losses <= ACCEPTABLE_LOSS_TOLERANCE)
    assert num_models_to_aggregate <= MAX_MODELS_TO_TAKE_PER_MSA
    print('Found %d best fit models within threshold for %s' % (num_models_to_aggregate, MSAS_TO_PRETTY_NAMES[msa_name]))

    # Aggregate predictions from best fit models that are within the ACCEPTABLE_LOSS_TOLERANCE
    mdl_predictions = []
    old_projected_hrs = None
    individual_plotting_kwargs = plotting_kwargs.copy()
    individual_plotting_kwargs['return_mdl_pred_and_hours'] = True  # don't plot individual models
    for model_idx in range(num_models_to_aggregate):
        ts = subdf.iloc[model_idx]['timestring']
        model, kwargs, _, _, _ = load_model_and_data_from_timestring(
            ts,
            load_fast_results_only=False, old_directory=old_directory, load_full_model=True)
        model_kwargs = kwargs['model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        mdl_prediction, projected_hrs = plot_model_fit_from_model_and_kwargs(
            ax,
            model_kwargs,
            data_kwargs,
            model=model,
            plotting_kwargs=individual_plotting_kwargs,
            train_test_partition=train_test_partition)
        mdl_predictions.append(mdl_prediction)
        if old_projected_hrs is not None:
            assert projected_hrs == old_projected_hrs
        old_projected_hrs = projected_hrs
    mdl_predictions = np.concatenate(mdl_predictions, axis=0)

    # Plot aggregate predictions
    agg_plotting_kwargs = plotting_kwargs.copy()
    agg_plotting_kwargs['mdl_prediction'] = mdl_predictions
    agg_plotting_kwargs['projected_hrs'] = projected_hrs
    plot_model_fit_from_model_and_kwargs(
        ax,
        model_kwargs,
        data_kwargs,
        plotting_kwargs=agg_plotting_kwargs,
        train_test_partition=train_test_partition,
    )
    ax.grid(False)

def plot_best_models_fit_for_all_msas(df, df_str, train_test_partition, key_to_sort_by, 
                                      thing_to_plot, plot_daily_not_cumulative, save_fig=False,
                                      y_maxes=None):
    if train_test_partition is None:
        train_test_partition_str = 'full_fit'
    else:
        train_test_partition_str = 'oos_fit'
    prefix_to_save_plot_with = f'trajectory_2x5_grid_{thing_to_plot}_{train_test_partition_str}_{df_str}'
    print(prefix_to_save_plot_with)
    fig, axes = plt.subplots(2, 5, figsize=(20,7), sharex=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes_list = [ax for axes_row in axes for ax in axes_row]
    for ax_idx, (ax, msa_name) in enumerate(zip(axes_list, MSAS)):
        if thing_to_plot == 'cases':
            x_start = datetime.datetime(2020, 3, 8)
            real_data_color = 'tab:orange'
            model_color = 'tab:blue'
            model_line_label = 'Model predictions'
            true_line_label = 'Reported cases'                 
        elif thing_to_plot == 'deaths':
            x_start = datetime.datetime(2020, 3, 19)
            real_data_color = 'tab:brown'
            model_color = 'tab:olive'
            model_line_label = 'Model predictions'
            true_line_label = 'Reported deaths'

        if train_test_partition_str == 'oos_fit':
            xticks = [x_start, 
                      datetime.datetime(2020, 4, 15),
                      datetime.datetime(2020, 5, 9)]
        else:
            xticks = [x_start,                               
                      datetime.datetime(2020, 5, 9)]   
        if y_maxes is not None:
            y_range = (0, y_maxes[ax_idx])
        else:
            y_range = 0

        other_plotting_kwargs = {
                'plot_log':False, 
                'plot_legend':False,
                'plot_errorbars':True,
                'xticks':xticks,
                'x_range':[x_start,
                           datetime.datetime(2020, 5, 9)],
                'y_range':y_range,
                'plot_daily_not_cumulative':plot_daily_not_cumulative,
                'plot_mode':thing_to_plot,
                'title_fontsize':20,
                'marker_size':4,
                'model_line_label': model_line_label,
                'true_line_label': true_line_label,
                'real_data_color':real_data_color,
                'model_color':model_color,
                'only_two_yticks':True,
        }
        plot_best_models_fit_for_msa(df, msa_name, ax, key_to_sort_by, train_test_partition, 
                         other_plotting_kwargs)
        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(-0.4, 1.04), fontsize=14)                              

    if save_fig:
        plt.savefig('covid_figures_for_paper/trajectories/%s.svg' % prefix_to_save_plot_with, 
                    bbox_inches='tight',
                    dpi=600)
        
def make_param_plausibility_plot(r0_df, make_r0_base_plot=True, make_r0_poi_plot=True, 
                                 max_beta=0.05, max_psi=15000, 
                                 min_r0_base=0.1, max_r0_base=2,
                                 min_r0_poi=1, max_r0_poi=3, 
                                 make_rainbow_plot=True):
    all_ranges = {}
    r0_df = r0_df.loc[(r0_df['home_beta'] <= max_beta) & (r0_df['poi_psi'] <= max_psi)].copy()
    r0_df['MSA'] = r0_df['MSA_name'].map(lambda x:MSAS_TO_PRETTY_NAMES[x])
    fontsize = 16

    if make_rainbow_plot:
        fig = plt.figure(figsize=[9, 8])
        n_rows = 2
    else:
        fig = plt.figure(figsize=[8, 4])
        n_rows = 1
    
    if make_r0_base_plot:
        ax = fig.add_subplot(n_rows, 2, 1)
        r0_base = (r0_df[['home_beta', 'R0_base']]
         .groupby(['home_beta'])
         .agg(['min', 'mean', 'max', 'size'])).reset_index()
        r0_base.columns = ['home_beta', 'min', 'mean', 'max', 'n_models_fit']
        ax.plot(r0_base['home_beta'],
                r0_base['mean'])
        ax.set_xlim([r0_base['home_beta'].min(), r0_base['home_beta'].max()])
        ax.set_xlabel(r"$\beta_{base}$", fontsize=fontsize)
        ax.set_ylabel("$R_{base}$", fontsize=fontsize)
        ax.plot(ax.get_xlim(), [min_r0_base, min_r0_base], linestyle='--', color='black')
        ax.plot(ax.get_xlim(), [max_r0_base, max_r0_base], linestyle='--', color='black')
        ax.set_ylim([0, max(ax.get_ylim())])
        slope, intercept, r_value, p_value, std_err = linregress(
            r0_base['home_beta'],
            r0_base['mean'])
        assert r_value > .99
        print("R0_base lies in plausible range (%2.5f-%2.5f) for home_beta %2.5f-%2.5f" % (
                min_r0_base,
                max_r0_base,
                min_r0_base/slope,
                max_r0_base/slope))
        all_ranges['min_home_beta'] = min_r0_base/slope
        all_ranges['max_home_beta'] = max_r0_base/slope
        if make_rainbow_plot:
            level_order = list(r0_df.loc[r0_df['poi_psi'] == 1000, ['R0_POI', 'MSA']].sort_values(by='R0_POI')['MSA'])[::-1]
            ax = fig.add_subplot(2, 2, 3)
            g = sns.lineplot(ax=ax, data=r0_df, x='home_beta', y='R0_base', hue='MSA', palette='rainbow', legend=False, hue_order=level_order)
            ax.set_xlim([r0_base['home_beta'].min(), r0_base['home_beta'].max()])
            ax.set_xlabel(r"$\beta_{base}$", fontsize=fontsize)
            ax.set_ylabel("$R_{base}$", fontsize=fontsize)
            ax.plot(ax.get_xlim(), [min_r0_base, min_r0_base], linestyle='--', color='black')
            ax.plot(ax.get_xlim(), [max_r0_base, max_r0_base], linestyle='--', color='black')
            ax.set_ylim([0, max(ax.get_ylim())])



    # poi_psi plot
    if make_r0_poi_plot:
        ax = fig.add_subplot(n_rows, 2, 2)
        r0_poi = (r0_df[['poi_psi', 'R0_POI']]
         .groupby(['poi_psi'])
         .agg(['min', 'mean', 'max', 'size'])).reset_index()
        r0_poi.columns = ['poi_psi', 'min', 'mean', 'max', 'n_models_fit']
        print(r0_poi)

        ax.plot(r0_poi['poi_psi'],
                r0_poi['mean'])
        ax.fill_between(r0_poi['poi_psi'], r0_poi['min'], r0_poi['max'], color='black', alpha=.2, label='range across MSAs')
        ax.set_xlim([r0_poi['poi_psi'].min(), r0_poi['poi_psi'].max()])
        ax.set_xlabel("$\psi$", fontsize=fontsize)
        ax.set_ylabel("$R_{POI}$", fontsize=fontsize)
        ax.plot(ax.get_xlim(), [min_r0_poi, min_r0_poi], linestyle='--', color='black', label='plausible range')
        ax.plot(ax.get_xlim(), [max_r0_poi, max_r0_poi], linestyle='--', color='black')
        ax.legend( loc='upper left')
        ax.set_ylim([0, max(ax.get_ylim())])

        slope, intercept, r_value, p_value, std_err = linregress(r0_poi['poi_psi'], r0_poi['min'])
        assert r_value > .99
        print("R0_POI min lies in plausible range (%2.5f-%2.5f) for poi_psi %2.5f-%2.5f" % (
                min_r0_poi,
                max_r0_poi,
                min_r0_poi/slope,
                max_r0_poi/slope))
        all_ranges['max_poi_psi'] = max_r0_poi/slope

        slope, intercept, r_value, p_value, std_err = linregress(r0_poi['poi_psi'], r0_poi['max'])
        assert r_value > .99
        print("R0_POI max lies in plausible range (%2.5f-%2.5f) for poi_psi %2.5f-%2.5f" % (
                min_r0_poi,
                max_r0_poi,
                min_r0_poi/slope,
                max_r0_poi/slope))
        all_ranges['min_poi_psi'] = min_r0_poi/slope

        if make_rainbow_plot:
            ax = fig.add_subplot(2, 2, 4)
            g = sns.lineplot(ax=ax, data=r0_df, x='poi_psi', y='R0_POI', hue='MSA', palette='rainbow', hue_order=level_order)
            ax.set_xlim([r0_poi['poi_psi'].min(), r0_poi['poi_psi'].max()])
            ax.set_xlabel("$\psi$", fontsize=fontsize)
            ax.set_ylabel("$R_{POI}$", fontsize=fontsize)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.plot(ax.get_xlim(), [min_r0_poi, min_r0_poi], linestyle='--', color='black', label='plausible range')
            ax.plot(ax.get_xlim(), [max_r0_poi, max_r0_poi], linestyle='--', color='black')
    fig.subplots_adjust(wspace=0.3)
    plt.savefig('covid_figures_for_paper/param_plausibility_plot.pdf', bbox_inches='tight')
    return all_ranges

###################################################################################
# Figure 2: mobility reduction and reopening analysis
###################################################################################
def get_daily_ts(poi_cbg_visits_list):
    """
    Sum over hourly counts to get total daily visits.
    """
    daily_ts = []
    for i in np.arange(0, len(poi_cbg_visits_list), 24):
        day_total = 0
        for j in range(i, i+24):
            day_total += poi_cbg_visits_list[j].sum()
        daily_ts.append(day_total)
    return daily_ts

def make_schematic(orig_visits, lesser_extent, shifted, colors, ax):
    days = helper.list_datetimes_in_range(MIN_DATETIME, MAX_DATETIME)
    daily_ts = get_daily_ts(orig_visits)
    ax.plot_date(days, daily_ts, linestyle='-', marker='.', color=colors[0], label='actual',
                linewidth=3)
    daily_ts = get_daily_ts(lesser_extent)
    ax.plot_date(days, daily_ts, linestyle='-', marker='.', color=colors[1], label='50% of actual',
                linewidth=3, alpha=0.5)
    daily_ts = get_daily_ts(shifted)
    ax.plot_date(days, daily_ts, linestyle='-', marker='.', color=colors[2], label='7 days later',
                linewidth=3, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.SU, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.legend(fontsize=16)
    ax.grid(alpha=0.3)
    ax.set_ylabel('Total POI visits per day', fontsize=18)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_title('Examples of modified mobility data', fontsize=20)
    ax.tick_params(labelsize=16)

def plot_lir_over_time_for_multiple_models(timestrings, ax, label, color, return_CI=False):
    """
    Given a list of timestrings, plot the fraction of people in states L, I, or R over time, 
    along with confidence intervals.
    """
    all_lir = []
    hours = None
    for ts in timestrings:
        model, kwargs, _, _, _ = load_model_and_data_from_timestring(ts, load_fast_results_only=False,
                                                                             load_full_model=True)

        new_hours = [kwargs['model_kwargs']['min_datetime'] + datetime.timedelta(hours=a)
                 for a in range(model.history['all']['latent'].shape[1])]
        if hours is None:
            hours = new_hours
        else:
            assert list(hours) == list(new_hours) # make sure hours stays unchanged between timestrings.
        lir = (model.history['all']['latent'] +
           model.history['all']['infected'] +
           model.history['all']['removed']) / model.history['all']['total_pop']
        all_lir.append(lir)
    all_lir = np.concatenate(all_lir, axis=0)
    print('Num params x seeds:', all_lir.shape[0])
    mean = INCIDENCE_POP * np.mean(all_lir, axis=0)
    lower_bound = INCIDENCE_POP * np.percentile(all_lir, LOWER_PERCENTILE, axis=0)
    upper_bound = INCIDENCE_POP * np.percentile(all_lir, UPPER_PERCENTILE, axis=0)
    ax.plot_date(hours, mean, linestyle='-', label=label, color=color)
    ax.fill_between(hours, lower_bound, upper_bound, alpha=0.5, color=color)
    ax.set_xlim([min(hours), max(hours)])
    if return_CI:
        return mean, lower_bound, upper_bound
    return mean

def make_counterfactual_line_plots(counterfactual_df, msa, ax, mode,
                                   cmap_str='viridis', y_lim=None, return_CI=False):
    assert mode in {'degree', 'shift-later', 'shift-earlier', 'shift'}
    if mode == 'degree':
        colors = list(cm.get_cmap(cmap_str, 5).colors)
        colors.reverse()
        values = [0, .25, .5, np.nan]  # put highest curve first, so that legend order is correct
        param_name = 'counterfactual_distancing_degree'
        subtitle = 'Magnitude of mobility reduction'
    else:
        colors = list(cm.get_cmap(cmap_str, 6).colors)
        colors.reverse()
        colors = colors[1:]
        subtitle = 'Timing of mobility reduction'
        param_name = 'counterfactual_shift_in_days'
        if mode == 'shift-later':
            values = [14, 7, 3, np.nan]
        elif mode == 'shift-earlier':
            values = [np.nan, -3, -7, -14]
            colors = colors[3:]  # so that true curve maintains the same color
        else:
            values = [7, 3, np.nan, -3, -7]

    msa_df = counterfactual_df[counterfactual_df['MSA_name'] == msa]
    color_idx = 0
    means = []
    lower_bounds = []
    upper_bounds = []
    for i, val in enumerate(values):
        if np.isnan(val):
            # plot the best-fit models for comparison. counterfactual_baseline_model provides the timestrings of the best-fit models to actual data.
            timestrings = msa_df.counterfactual_baseline_model.unique()
            label = 'actual' if mode == 'degree' else '0 days (actual)'
            mean, lower_bound, upper_bound = plot_lir_over_time_for_multiple_models(timestrings, ax, label, 'black', return_CI=True)
        else:
            # plot the models from this experiment
            msa_val_df = msa_df[msa_df[param_name] == val]
            timestrings = msa_val_df.timestring.values
            if mode == 'degree':
                label = 'no reduction' if val == 0 else '%d%% of actual' % (val * 100)
            else:  # mode is some kind of shift
                postfix = 'earlier' if val < 0 else 'later'
                label = '%d days %s' % (abs(val), postfix)  # take abs value in case val is negative
            mean, lower_bound, upper_bound = plot_lir_over_time_for_multiple_models(timestrings, ax, label, colors[i], return_CI=True)
        means.append(mean)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    ax.legend(loc='upper left', fontsize=16)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Cumulative infections (per 100k)', fontsize=18)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(alpha=.5)
    ax.tick_params(labelsize=16)
    ax.set_title(subtitle, fontsize=20)
    if return_CI:
        return means, lower_bounds, upper_bounds, colors[:len(means)]
    return means, colors[:len(means)]

def get_final_LIR_fraction_for_multiple_models(timestrings):
    """
    Get the final fraction of people in state L+I+R; returns vector of length [seeds x params].
    """
    final_frac = []
    for ts in timestrings:
        model, kwargs, _, _, _ = load_model_and_data_from_timestring(ts, load_fast_results_only=False,
                                                                     load_full_model=True)
        all_history = model.history['all']
        all_lir = all_history['latent'] + all_history['infected'] + all_history['removed']
        frac_in_lir = all_lir / all_history['total_pop']
        final_frac.extend(frac_in_lir[:, -1])

    assert (len(np.array(final_frac).shape) == 1) and (len(final_frac) % len(timestrings) == 0)
    return np.array(final_frac)

def get_ratios_to_baselines(msa_df, param_name, param_values, baselines, baseline_timestrings):
    cols = []
    all_ratios = []
    timestrings = None
    for i, val in enumerate(param_values):
        msa_val_df = msa_df[msa_df[param_name] == val]
        timestrings = msa_val_df.timestring.values
        assert len(set(timestrings)) == len(timestrings)

        # sanity check to make sure we are comparing to the same baselines.
        assert list(baseline_timestrings) == list(msa_val_df.counterfactual_baseline_model)
        final_fracs = get_final_LIR_fraction_for_multiple_models(timestrings)
        ratios = final_fracs / baselines
        cols.append('%s=%s' % (param_name.strip('counterfactual_'), val))
        mean = np.mean(ratios)
        lower_bound = np.percentile(ratios, LOWER_PERCENTILE)
        upper_bound = np.percentile(ratios, UPPER_PERCENTILE)
        all_ratios.append((round(mean, 3), (round(lower_bound, 3), round(upper_bound, 3))))
    return cols, all_ratios

def get_counterfactual_ratios_at_datetime(counterfactual_df, msa):
    """
    How many more infections do we get under counterfactuals than we actually got?
    """
    msa_df = counterfactual_df[counterfactual_df['MSA_name'] == msa].copy()
    # important to sort to make sure that order of baselines remains consistent with the models we compare to.
    msa_df = msa_df.sort_values(by='counterfactual_baseline_model')
    baseline_timestrings = sorted(msa_df.counterfactual_baseline_model.unique())
    baselines = get_final_LIR_fraction_for_multiple_models(baseline_timestrings)

    param_name = 'counterfactual_distancing_degree'
    param_values = [0, .25, .5]
    cols, ratios = get_ratios_to_baselines(msa_df, param_name, param_values, baselines, baseline_timestrings)
    param_name = 'counterfactual_shift_in_days'
    param_values = [7, 3, -3, -7]
    cols2, ratios2 = get_ratios_to_baselines(msa_df, param_name, param_values, baselines, baseline_timestrings)
    cols.extend(cols2)
    ratios.extend(ratios2)
    return cols, ratios

def get_poi_densities(poi_characteristics, poi_cbg_visits_list):
    """
    Get average density of people in a POI.
    """
    # extract the number of visits to the POI each hour.
    num_pois, num_cbgs = poi_cbg_visits_list[0].shape
    assert num_pois == len(poi_characteristics['poi_areas'])
    visits_to_poi_in_hour = []
    ones_vector = np.ones(num_cbgs)  # just a faster way to do sums.
    for poi_cbg_visits in poi_cbg_visits_list:
        visits_to_poi_in_hour.append(poi_cbg_visits @ ones_vector)
    visits_to_poi_in_hour = np.array(visits_to_poi_in_hour)  # n_hours x n_pois
    assert visits_to_poi_in_hour.shape == (len(poi_cbg_visits_list), num_pois)

    # compute proportion of visits by hour
    total_visits = visits_to_poi_in_hour.sum(axis=0)
    total_visits[total_visits == 0] = 1 # avoid divide by zero problems
    proportion_visits_to_poi_in_hour = visits_to_poi_in_hour / total_visits

    # some sanity checks
    assert proportion_visits_to_poi_in_hour.shape == visits_to_poi_in_hour.shape
    assert np.isnan(proportion_visits_to_poi_in_hour).sum() == 0
    summed_props =  proportion_visits_to_poi_in_hour.sum(axis=0)
    assert np.allclose(summed_props[summed_props != 0], 1)
    assert (summed_props == 0).mean() < 0.01

    # weighted number of visitors / area
    weighted_visits_over_area = ((proportion_visits_to_poi_in_hour * visits_to_poi_in_hour).sum(axis=0)/
                         poi_characteristics['poi_areas'])
    squared_visits_over_area = ((visits_to_poi_in_hour * visits_to_poi_in_hour).sum(axis=0)/
                         poi_characteristics['poi_areas'])
    return weighted_visits_over_area, squared_visits_over_area

def make_superspreader_plot_for_msa(df, msa, ax, plot_log=False, set_labels=True,
                                    poi_and_cbg_characteristics=None, color='tab:blue',
                                    line_label=None, linestyle='solid'):
    """
    Produce a CDF of what fraction of POI infections are accounted for by the most infectious POIs.
    """
    city_df = df.loc[df['MSA_name'] == msa]
    all_poi_counts_for_city = []
    city_proportions_of_total_infections_from_pois = []
    for i in range(len(city_df)):
        timestring = city_df.iloc[i]['timestring']
        mdl, kwargs, _, _, _ = load_model_and_data_from_timestring(
                timestring,
                load_fast_results_only=False,
                load_full_model=True)
        city_proportions_of_total_infections_from_pois += list((mdl.history['all']['new_cases_from_poi'].sum(axis=1)/
                                mdl.history['all']['new_cases'].sum(axis=1)))
        min_datestring = kwargs['model_kwargs']['min_datetime'].strftime('%B %-d')
        max_datestring = kwargs['model_kwargs']['max_datetime'].strftime('%B %-d')
        num_cases_per_poi = mdl.history['all']['num_cases_per_poi']
        if len(num_cases_per_poi.shape) == 3:  # seed x poi x time
            num_cases_per_poi = np.sum(num_cases_per_poi, axis=2)  # sum over time
        all_poi_counts_for_city.append(num_cases_per_poi)
        if len(all_poi_counts_for_city) > 1:
            assert all_poi_counts_for_city[-1].shape == all_poi_counts_for_city[-2].shape

    prop_from_pois_df = pd.DataFrame({'msa':msa,
                                      'prop_total_infections_from_pois':city_proportions_of_total_infections_from_pois})
    # all_poi_counts_for_city is n_seeds x n_pois
    all_poi_counts_for_city = np.concatenate(all_poi_counts_for_city, axis=0)
    all_poi_fracs_for_city = all_poi_counts_for_city/all_poi_counts_for_city.sum(axis=1).reshape([len(all_poi_counts_for_city), 1])
    assert np.allclose(all_poi_fracs_for_city.sum(axis=1), 1)
    mean_frac_per_poi = np.mean(all_poi_fracs_for_city, axis=0)

    sorted_idxs = np.argsort(mean_frac_per_poi)[::-1]
    all_poi_fracs_for_city = all_poi_fracs_for_city[:, sorted_idxs]
    cumulative_poi_fracs = np.cumsum(all_poi_fracs_for_city, axis=1)
    x = np.linspace(0, 1, cumulative_poi_fracs.shape[1])
    lower_CI = np.percentile(cumulative_poi_fracs, LOWER_PERCENTILE, axis=0)
    upper_CI = np.percentile(cumulative_poi_fracs, UPPER_PERCENTILE, axis=0)

    if line_label is None:
        ax.plot(x, cumulative_poi_fracs.mean(axis=0), color=color, linestyle=linestyle)
    else:
        ax.plot(x, cumulative_poi_fracs.mean(axis=0), color=color, label=line_label, linestyle=linestyle)
    ax.fill_between(x, lower_CI, upper_CI, color=color, alpha=.2)
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([1e-4, 1])
    if plot_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ticks = np.arange(0.0, 1.01, .1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        tick_labels = ['%d%%' % (t * 100) for t in ticks]
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    ax.grid(alpha=.3)
    ax.tick_params(labelsize=16)
    if set_labels:
        ax.set_xlabel("Percent of POIs", fontsize=18)
        ax.set_ylabel("Percent of POI infections", fontsize=18)
        ax.set_title('Cumulative distribution of\npredicted infections over POIs', fontsize=20)

    # compute correlation btwn POI's infectiousness and its characteristics
    if poi_and_cbg_characteristics is not None:
        poi_characteristics = poi_and_cbg_characteristics[msa]
        fn = get_ipf_filename(msa, MIN_DATETIME, MAX_DATETIME, True, True)
        print(fn)
        f = open(fn, 'rb')
        poi_cbg_visits_list = pickle.load(f)
        f.close()
        weighted_visits_over_area, squared_visits_over_area = get_poi_densities(poi_characteristics, poi_cbg_visits_list)
        poi_characteristics_df = pd.DataFrame({'mean_frac_of_infections_at_poi':mean_frac_per_poi,
             'density*dwell_time_factor':poi_characteristics['poi_dwell_time_correction_factors'] * weighted_visits_over_area,
             'visits^2*dwell_time_factor/area':poi_characteristics['poi_dwell_time_correction_factors'] * squared_visits_over_area,
             'weighted_visits_over_area':weighted_visits_over_area,
             'weighted_visits':poi_characteristics['poi_areas'] * weighted_visits_over_area,
             'dwell_time':poi_characteristics['poi_dwell_times']})
        cutoff_90 = scoreatpercentile(poi_characteristics_df['mean_frac_of_infections_at_poi'], 90)
        cutoff_99 = scoreatpercentile(poi_characteristics_df['mean_frac_of_infections_at_poi'], 99)
        poi_characteristics_df['infectiousness_group'] = poi_characteristics_df['mean_frac_of_infections_at_poi'].map(lambda x:'top 10%' if x >= cutoff_90 else 'bottom 90%')
        print("Spearman correlations (across POIs) between POI characteristics and fraction of total infections at POI")
        print(poi_characteristics_df.corr(method='spearman')['mean_frac_of_infections_at_poi'])
        print(poi_characteristics_df.groupby('infectiousness_group').median().transpose()[['bottom 90%', 'top 10%']])
    return prop_from_pois_df, all_poi_fracs_for_city

def get_pareto_curve(results_df, msa, param_name, intervention_idx, get_diff_in_infections,
                     full_activity_num_visits=None, cbg_group='all'):
    """
    Returns the "pareto curve" of this reopening strategy, i.e., the tradeoff between the percent of POI 
    visits lost vs the number of infections inccurred.
    """
    msa_df = results_df[results_df['MSA_name'] == msa]
    values = sorted(msa_df[param_name].unique())  # parameter for reopening strategy, eg degree of uniform reduction
    X = []  # prop visits lost (relative to full reopening)
    # if get_diff_in_infections is True, Y is the increase in infections after reopening; 
    # otherwise, it is the cumulative number of infections at the end of the simulation
    Y_mean = []  # mean over models and seeds
    Y_min = []  # lower-bound on CI for y
    Y_max = []  # upper-bound on CI for y
    all_intervention_lir = []  # num infections at the point of reopening

    for i, val in enumerate(values):
        msa_val_df = msa_df[msa_df[param_name] == val]
        timestrings = msa_val_df.timestring.values
        visits_lost = None
        final_lir = [] 
        for ts in timestrings:
            curr_intervention_lir, curr_final_lir, curr_visits_lost = get_lir_checkpoints_and_prop_visits_lost(ts, 
                    intervention_idx, group=cbg_group, full_activity_num_visits=full_activity_num_visits)
            all_intervention_lir.extend(list(curr_intervention_lir))
            if get_diff_in_infections:
                final_lir.extend(list(curr_final_lir - curr_intervention_lir))
            else:  # plot all infections, not just those gained
                final_lir.extend(list(curr_final_lir))
            if visits_lost is None:
                visits_lost = curr_visits_lost
            else:  # visits_lost should be the same for every model with this parameter value
                assert visits_lost == curr_visits_lost
        X.append(visits_lost)
        Y_min.append(INCIDENCE_POP * np.percentile(final_lir, LOWER_PERCENTILE))
        Y_mean.append(INCIDENCE_POP * np.mean(final_lir))
        Y_max.append(INCIDENCE_POP * np.percentile(final_lir, UPPER_PERCENTILE))
        if i == 0:
            print('Num params * seeds:', len(final_lir))
    return X, Y_min, Y_mean, Y_max, all_intervention_lir, values

def make_pareto_plot(X, Y_min, Y_mean, Y_max, ax, all_intervention_lir=None,
                     color=None, point_labels=None, annotation_color=None,
                     line_label=None, set_axis_labels=True, return_data=False, xytext_offset=(0,5)):    
    """
    Plots the pareto curve. Most of the kwargs just control various aspects of plot appearance.
    """
    if line_label:
        ax.plot(X, Y_mean, marker='o', linewidth=2, color=color, label=line_label)
    else:
        ax.plot(X, Y_mean, marker='o', linewidth=2, color=color)
    ax.fill_between(X, Y_min, Y_max, alpha=0.3, color=color)
    if point_labels is not None:
        assert annotation_color is not None
        for val, x, y in zip(point_labels, X, Y_mean):
            if val < .8:
                ax.annotate('%d%%' % (100 * val), xy=(x, y), xytext=xytext_offset, textcoords='offset pixels', fontsize=16, color=annotation_color)
    
    # plot the average num infections at the point of reopening
    if all_intervention_lir is not None:  
        intervention_lir_mean = np.mean(all_intervention_lir)
        ax.plot([min(X), max(X)], [INCIDENCE_POP * intervention_lir_mean, INCIDENCE_POP * intervention_lir_mean], color='red', linestyle='--', label='cumulative infections\nbefore reopening')
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    
    if set_axis_labels:
        ax.set_xlabel('Fraction of visits lost from partial reopening\n(compared to full reopening)', fontsize=18)
        if all_intervention_lir is None:  
            ax.set_ylabel('New infections (per 100k)\nin month after reopening', fontsize=18)
        else:
            ax.set_ylabel('Cumulative infections (per 100k)', fontsize=18)
        ax.set_title('Capping hourly visits at x% of\nPOI maximum occupancy', fontsize=20)
        ax.legend(fontsize=16)
    ax.grid(alpha=.3)
    if np.max(X) > 1000:
        ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax.tick_params(labelsize=16)
    if return_data:
        return point_labels, X, Y_mean, Y_min, Y_max

def plot_pairwise_comparison(max_cap_df, uniform_df, msa_name, full_activity_num_visits, intervention_idx,
                             ax, cbg_group='all', mode='ratio', color='slategrey', line_label=None, x_lim=None,
                             yticks=None, set_axis_labels=True):
    """
    Plots the pairwise comparison between two partial reopening experiments: clipping and uniform reduction.
    They are compared on the basis of the number of infections gained post-reopening, and may be compared 
    in terms of ratio, difference, or percent change.
    """
    assert mode in {'ratio', 'diff', 'percent_change'}
    msa_mc_df = max_cap_df[max_cap_df['MSA_name'] == msa_name]
    max_cap_vals = sorted(msa_mc_df.counterfactual_max_capacity_alpha.unique())
    msa_u_df = uniform_df[uniform_df['MSA_name'] == msa_name]
    uniform_vals = sorted(msa_u_df.counterfactual_full_activity_alpha.unique())
    print('Found %d rows for MSA in max cap and %d rows for MSA in uniform' % (len(msa_mc_df), len(msa_u_df)))
    assert len(max_cap_vals) == len(uniform_vals)
    assert math.isclose(max_cap_vals[-1], 1.0)
    assert math.isclose(uniform_vals[-1], 1.0)
    X = []  # x-axis: prop visits lost (relative to full reopening)
    Y_mean = []  # y-axis: comparison in number of infection gained post-reopening
    Y_lower = []  # lower-bound on CI for comparison
    Y_upper = []  # upper-bound on CI for comparison
    for mc_val, u_val in zip(max_cap_vals, uniform_vals):
        # each pair of mc_val, u_val [which evaluate one max capacity and one uniform reduction policy, respectively] should be matched on proportion of visits lost. 
        print('Comparing max_cap_alpha=%.2f to full_activity_alpha=%.4f' % (mc_val, u_val))
        mc_subdf = msa_mc_df[msa_mc_df['counterfactual_max_capacity_alpha'] == mc_val]
        u_subdf = msa_u_df[msa_u_df['counterfactual_full_activity_alpha'] == u_val]
        mc_all_lir = []
        u_all_lir = []
        visits_lost = None
        for baseline_model in mc_subdf.counterfactual_baseline_model.values:  # per parameter setting
            curr_df = mc_subdf[mc_subdf['counterfactual_baseline_model'] == baseline_model]
            assert len(curr_df) == 1
            int_lir, fin_lir, mc_visits_lost = get_lir_checkpoints_and_prop_visits_lost(curr_df.iloc[0]['timestring'], 
                                                                                     intervention_idx, group=cbg_group,
                                                                                     full_activity_num_visits=full_activity_num_visits)
            frac_gained = fin_lir - int_lir  # list of infections gained, per seed
            mc_all_lir.extend(list(frac_gained.copy()))
            
            curr_df = u_subdf[u_subdf['counterfactual_baseline_model'] == baseline_model]
            assert len(curr_df) == 1
            int_lir, fin_lir, u_visits_lost = get_lir_checkpoints_and_prop_visits_lost(curr_df.iloc[0]['timestring'], 
                                                                                     intervention_idx, group=cbg_group)
            frac_gained = fin_lir - int_lir 
            u_all_lir.extend(list(frac_gained.copy()))
            assert abs(mc_visits_lost - u_visits_lost) < .01, (mc_visits_lost, u_visits_lost)  # they should be matched on proportion of visits lost
            visits_lost = mc_visits_lost
        print('Num params * seeds:', len(mc_all_lir))
        if mode == 'ratio':
            curr_comparison = np.array(mc_all_lir) / np.array(u_all_lir)
        elif mode == 'diff':
            curr_comparison = np.array(mc_all_lir) - np.array(u_all_lir)
        else:
            curr_comparison = (np.array(mc_all_lir) - np.array(u_all_lir)) / np.array(u_all_lir)
        X.append(visits_lost)
        Y_mean.append(np.mean(curr_comparison))
        Y_lower.append(np.percentile(curr_comparison, LOWER_PERCENTILE))
        Y_upper.append(np.percentile(curr_comparison, UPPER_PERCENTILE))
    print('X', X)
    print('Y lower', Y_lower)
    print('Y mean', Y_mean)
    print('Y upper', Y_upper)
    if line_label is None:
        ax.plot(X, Y_mean, marker='o', linewidth=2, color=color)
    else:
        ax.plot(X, Y_mean, marker='o', linewidth=2, color=color, label=line_label)
    ax.fill_between(X, Y_lower, Y_upper, alpha=0.2, color=color)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if yticks is not None:
        ax.set_ylim((np.min(yticks), np.max(yticks)))
        ax.set_yticks(yticks)
    if mode == 'ratio':
        ax.plot([min(X), max(X)], [1, 1], color='grey', linestyle='--')
        ylabel = 'Ratio of new infections'
    else:
        ax.plot([min(X), max(X)], [0, 0], color='grey', linestyle='--')
        if mode == 'diff':
            ylabel = 'Difference in new infections'
        else:
            ylabel = 'Relative change in new infections'
    if set_axis_labels:
        ax.set_xlabel('Fraction of visits lost from partial reopening\n(compared to full reopening)', fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title('Change in new infections with reduced occupancy\nreopening instead of uniform reduction', fontsize=20) 
    ax.grid(alpha=.3)
    ax.tick_params(labelsize=16)
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_decimal_as_percent))
    return X, Y_mean, Y_lower, Y_upper

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

def make_boxplot_of_poi_reopening_effects(intervention_df, msa_names, poi_and_cbg_characteristics, 
                                          titlestring, cats_to_plot, filename, only_plot_reopening_impact=False):
    """
    Make boxplots of the effects (on fraction infected) of opening each category of POI. 
    """
    assert len(msa_names) > 0
    print("Making plots using", msa_names)
    subcategory_counts = {}

    # each row in poi_characteristics_df is one POI. 
    poi_characteristics_df = []
    for msa in msa_names:
        print(msa)
        poi_characteristics = poi_and_cbg_characteristics[msa].copy()
        if 'poi_cbg_visits_list' in poi_characteristics:
            poi_cbg_visits_list = poi_characteristics['poi_cbg_visits_list']
        else:
            fn = get_ipf_filename(msa, MIN_DATETIME, MAX_DATETIME, True, True)
            print(fn)
            f = open(fn, 'rb')
            poi_cbg_visits_list = pickle.load(f)
            f.close()
        weighted_visits_over_area, squared_visits_over_area = get_poi_densities(poi_characteristics, poi_cbg_visits_list)
        msa_df = pd.DataFrame({'sub_category':poi_characteristics['poi_categories'],
                               'original_dwell_times':poi_characteristics['poi_dwell_times'],
                               'dwell_time_correction_factors':poi_characteristics['poi_dwell_time_correction_factors'],
                               'weighted_visits_over_area':weighted_visits_over_area,
                               'squared_visits_over_area':squared_visits_over_area})
        msa_df['pretty_name'] = msa_df['sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
        subcategory_counts[msa] = Counter(msa_df['pretty_name'])
        poi_characteristics_df.append(msa_df)

    poi_characteristics_df = pd.concat(poi_characteristics_df)
    poi_characteristics_df = poi_characteristics_df.loc[poi_characteristics_df['sub_category'].map(lambda x:x in cats_to_plot)]
    poi_characteristics_df['density*dwell_time_factor'] = poi_characteristics_df['dwell_time_correction_factors'] * poi_characteristics_df['weighted_visits_over_area']
    poi_characteristics_df['visits^2*dwell_time_factor/area'] = poi_characteristics_df['dwell_time_correction_factors'] * poi_characteristics_df['squared_visits_over_area']


    # each row in intervention_df is one fitted model. 
    intervention_df = intervention_df.copy()
    intervention_df = intervention_df.loc[intervention_df['MSA_name'].map(lambda x:x in msa_names)]
    total_modeled_pops = {} # number of people  we model in each MSA
    for msa in msa_names:
        ts = intervention_df.loc[intervention_df['MSA_name'] == msa, 'timestring'].iloc[0]
        model, _, _, _, _ = load_model_and_data_from_timestring(
                ts,
                load_fast_results_only=False,
                load_full_model=True)
        total_modeled_pops[msa] = model.CBG_SIZES.sum()


    intervention_df['pretty_cat_names'] = intervention_df['counterfactual_sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x in cats_to_plot)]
    full_reopen_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 1]
    full_close_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 0]
    merge_cols = ['model_fit_rank_for_msa', 'pretty_cat_names', 'MSA_name']

    # make each random seed into its own row. 
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
    combined_df['reopening_impact'] = combined_df['reopening_impact'] * INCIDENCE_POP
    print("Reopening impact quantifies cases per %i" % INCIDENCE_POP)

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
        print("Lower CI, additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .quantile(LOWER_PERCENTILE / 100)  # pandas quantile needs 0 < q <= 1
                       .sort_values(by='reopening_impact')[::-1].reset_index()))
        print("Upper CI, additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .quantile(UPPER_PERCENTILE / 100)
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


    # actually make box plots. 
    outlier_size = 1
    num_positive = np.sum(combined_df['reopening_impact'] > 0)
    print('%d / %d (num categories * seeds * model params) had reopening impact greater than 0' % (num_positive, len(combined_df)))
    print(combined_df.head())
    if not only_plot_reopening_impact:
        fig, axes = plt.subplots(2, 2, figsize=[15, 9])
        fig.subplots_adjust(wspace=10, hspace=0.5)
        for i, poi_characteristic in enumerate(['original_dwell_times', 'weighted_visits_over_area']):
            ax = axes[0][i]
            sns.boxplot(y="pretty_name",
                    x=poi_characteristic,
                    data=poi_characteristics_df,
                    order=list(mean_impact['pretty_cat_names']),
                    ax=ax,
                    fliersize=outlier_size)
            ax.set_ylabel("")
            if poi_characteristic == 'poi_areas':
                ax.set_xlabel("Area (sq feet)")
            elif poi_characteristic == 'original_dwell_times':
                ax.set_xlabel("Dwell time (minutes)")
                ax.set_xlim([0, 200])
            elif poi_characteristic == 'weighted_visits_over_area':
                ax.set_xlabel("Average visits per hour / sq ft")
                ax.set_xlim([1e-4, 1e-2])
            ax.grid(alpha=0.5)

        ax = axes[1][0]
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
        
        ax = axes[1][1]
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
        ax.set_xlim([10, 1e4])
        fig.suptitle(titlestring)
        plt.subplots_adjust(wspace=.6)
        ax.grid(alpha=0.5)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(9.5,7))
        sns.boxplot(y="pretty_cat_names",
                    x="reopening_impact",
                    data=combined_df,
                    order=mean_impact['pretty_cat_names'],
                    ax=ax,
                    whis=0,
                    fliersize=outlier_size)
        ax.set_xlabel("Additional infections (per 100k),\ncompared to not reopening", fontsize=18)
        ax.set_xscale('log')
        ax.set_ylabel("")
        ax.tick_params(labelsize=16)
        ax.set_xlim([10, 1e4])
        ax.set_title(titlestring, fontsize=20)
        plt.subplots_adjust(wspace=.6)
        ax.grid(alpha=0.5)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')    

def plot_reopening_effect_by_poi_category_with_disparate_impact(intervention_df, medians_or_deciles, cats_to_plot, filename=None):
    """
    Break down reopening impacts by top and bottom deciles. 
    While this plot is similar to that in make_boxplot_of_poi_reopening_effects, the code is unfortunately a bit different because
    of how we save models: we actually have to load in individual models to access results broken down for rich and poor. 
    """
    print("Analyzing results for rich and poor %s" % medians_or_deciles)
    assert medians_or_deciles in ['medians', 'deciles']
    intervention_df = intervention_df.copy()
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x in cats_to_plot)]
    intervention_df['mean_final_infected_fraction'] = intervention_df['final infected fraction'].map(lambda x:np.mean(x))
    assert len([a for a in REMOVED_SUBCATEGORIES if a in intervention_df['counterfactual_sub_category'].values]) == 0
    intervention_df = intervention_df.loc[intervention_df['counterfactual_sub_category'].map(lambda x:x not in REMOVED_SUBCATEGORIES)]
    intervention_df['pretty_cat_names'] = intervention_df['counterfactual_sub_category'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
    full_reopen_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 1]
    full_close_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 0]

    # make sure we have the expected numbers of models. 
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

    for setting in ['closed', 'open']:
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
                n_LIR = (mdl.history[g]['latent'] + mdl.history[g]['infected'] + mdl.history[g]['removed'])[:, -1] # take last timestep. First dimension is seeds. 
                results_by_group[g].append(n_LIR/n_group) # so each entry in the list results_by_group[g] is one set of random seeds for one model. 

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
    assert len(combined_df) == len(full_reopen_df)

    for group in ['top', 'bottom']:
        combined_df['impact_of_reopening_%s' % group] = (combined_df['%s_group_lir_if_opened' % group] - combined_df['%s_group_lir_if_closed' % group]) * INCIDENCE_POP
        combined_df['relative_impact_of_reopening_%s' % group] = combined_df['impact_of_reopening_%s' % group]/(combined_df['%s_group_lir_if_closed' % group] * INCIDENCE_POP)


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


    # order is a bit arbitrary; mean of impact_of_reopening_bottom across all random seeds (which does not weight MSAs equally). 
    # leaving for now because it isn't something we really focus on. 
    order_to_display_cats = (combined_df[['relative_impact_of_reopening_bottom',
                                       'relative_impact_of_reopening_top',
                                        'impact_of_reopening_bottom',
                                         'impact_of_reopening_top',
                                       'pretty_cat_names']].groupby('pretty_cat_names')
         .mean()
         .reset_index()
        .sort_values(by='impact_of_reopening_bottom'))
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
                # clip CIs that go outside the plot limits to avoid weirdness with log. Doesn't affect CIs inside the plot. 
                df_to_plot_for_msa['%s_group_lower_CI' % top_or_bottom].append(max(1, scoreatpercentile(small_df['impact_of_reopening_%s' % top_or_bottom], LOWER_PERCENTILE)))
                df_to_plot_for_msa['%s_group_upper_CI' % top_or_bottom].append(max(1, scoreatpercentile(small_df['impact_of_reopening_%s' % top_or_bottom], UPPER_PERCENTILE)))
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

###################################################################################
# Figure 3: equity analysis
###################################################################################
def get_LIR_ratios_from_models(timestrings, g1, g2):
    LIR_ratios = []
    for timestring in timestrings:
        mdl, _, _, _, _ = load_model_and_data_from_timestring(
            timestring,
            load_fast_results_only=False,
            load_full_model=True)

        ratios = []
        for idx, g in enumerate([g1, g2]):
            n_group = mdl.history[g]['total_pop']
            n_LIR = (mdl.history[g]['latent'] + mdl.history[g]['infected'] + mdl.history[g]['removed'])[:, -1]
            ratios.append(n_LIR / n_group)



        LIR_ratios.append(ratios[0] / ratios[1])
    LIR_ratios = np.array(LIR_ratios)
    assert len(LIR_ratios) == len(timestrings)
    return LIR_ratios

def make_all_disparities_infection_ratio_plots(df, key_to_sort_by='loss_dict_daily_cases_RMSE',
                                               loss_tolerance=ACCEPTABLE_LOSS_TOLERANCE, save_figs=False, return_all_ratios=False):
    table_results = []
    all_ratios = []
    print("Using parameters MAX_MODELS_TO_TAKE_PER_MSA=%i, loss_tolerance=%2.3f, and key_to_sort_by=%s" % 
          (MAX_MODELS_TO_TAKE_PER_MSA, loss_tolerance, key_to_sort_by))
    for msa in sorted(list(set(df['MSA_name']))):
        msa_df = df.loc[(df['MSA_name'] == msa)]
        min_loss = msa_df[key_to_sort_by].min()
        msa_df = (msa_df.loc[msa_df[key_to_sort_by] <= (min_loss * loss_tolerance)]
                   .sort_values(by=key_to_sort_by)
                   .iloc[:MAX_MODELS_TO_TAKE_PER_MSA])
        max_ratio = msa_df[key_to_sort_by].max() / msa_df[key_to_sort_by].min()
        assert max_ratio > 1 and max_ratio < loss_tolerance
        print("Plotting %i models for %s" % (len(msa_df), msa))
        for k in ['p_black', 'median_household_income', 'p_white']:
            for comparison in ['medians', 'deciles']:
                if comparison == 'medians':
                    disadvantaged_ratios = get_LIR_ratios_from_models(
                            msa_df['timestring'], 
                            f'{k}_below_median',
                            f'{k}_above_median')

                else:
                    disadvantaged_ratios = get_LIR_ratios_from_models(
                            msa_df['timestring'], 
                            f'{k}_bottom_decile',
                            f'{k}_top_decile')

                if k == 'p_black':
                    disadvantaged_ratios = 1 / disadvantaged_ratios

                # disadvantaged ratios here is n_models x n_seeds
                table_results.append({"MSA_name":msa, 
                                    'demo':k, 
                                    'comparison':comparison, 
                                    'median_ratio':np.median(disadvantaged_ratios),
                                    'disadvantaged_group_is_more_sick':np.mean(disadvantaged_ratios > 1), 
                                   'n':len(msa_df)})
                all_ratios.append(pd.DataFrame({'MSA_name':msa, 
                                                'ratio':disadvantaged_ratios.flatten(),
                                                'comparison':comparison, 
                                                'demo':k}))

    all_ratios = pd.concat(all_ratios)
    all_ratios['MSA'] = all_ratios['MSA_name'].map(lambda x:MSAS_TO_PRETTY_NAMES[x])
    table_results = pd.DataFrame(table_results)
    comparison = 'deciles'
    for demographic in ['p_white', 'median_household_income']:
        if save_figs:
            make_disparities_infection_ratio_plot_for_paper(all_ratios, 
                                                        comparisons=[comparison], 
                                                        demographic=demographic, 
                                                        filename='covid_figures_for_paper/infection_rate_disparities_%s_%s.pdf' 
                                                                % (demographic, comparison))
        else:
            make_disparities_infection_ratio_plot_for_paper(all_ratios, 
                                                        comparisons=[comparison], 
                                                        demographic=demographic, 
                                                        filename=None)
    if return_all_ratios:
        return table_results, all_ratios
    else:
        return table_results

def make_disparities_infection_ratio_plot_for_paper(all_ses_ratios, comparisons, demographic, filename):
    assert demographic in ['median_household_income', 'p_white']
    assert all([comparison in ['deciles', 'medians'] for comparison in comparisons])
    fig = plt.figure(figsize=[len(comparisons) * 7, 7])

    for ax_idx, comparison in enumerate(comparisons):
        ax = fig.add_subplot(1, len(comparisons), ax_idx + 1)
        rows_to_use = all_ses_ratios.loc[(all_ses_ratios['demo'] == demographic) & (all_ses_ratios['comparison'] == comparison)].copy()
        print("Median of medians for", demographic, comparison, rows_to_use.groupby('MSA')['ratio'].median().median())
        
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
                ax.set_title("Predicted disparities between CBGs\nin deciles with lowest/highest income", fontsize=20)
            else:
                ax.set_title("Predicted disparities between CBGs\nabove and below median income", fontsize=20)
            ax.set_xlabel("Relative infection risk of\nlower- to higher-income CBGs", fontsize=18)
        elif demographic == 'p_white':
            if comparison == 'deciles':
                ax.set_title("Predicted disparities between CBGs\nin deciles with lowest/highest % white", fontsize=20)
            else:
                ax.set_title("Predicted disparities between CBGs\nabove and below median for % white", fontsize=20)
            ax.set_xlabel("Relative infection risk of\nless white to more white CBGs", fontsize=18)
        else:
            raise Exception("Invalid demographic variable")

        plt.xticks([0.5, 1, 2, 5, 10, 20, 50], ['0.5x', '1x', '2x', '5x', '10x', '20x', '50x'])
        ax.tick_params(labelsize=16)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

def get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, 
                               poi_time_counts=None, group_to_track=None, verbose=True,
                               start_idx=None, end_idx=None, mode='normal', overall_risk=False):
    '''
    Creates a dataframe where each row represents a POI and contains its original
    POI index, static attributes of the POI (top_category, sub_category, area, dwell_time,
    dwell_time_correction_factor), and its time-varying attributes (avg_occupancy, avg_density,
    avg_transmission_rate) averaged over all hours and, if applicable, with respect to the CBGs
    specified in group_to_track.
    '''
    assert mode in {'normal', 'drop_area', 'drop_dwell_time', 'drop_visits',
                    'only_time_spent', 'only_density'}
    stuff = poi_and_cbg_characteristics[msa_name]
    poi_areas = stuff['poi_areas']
    poi_dwell_times = stuff['poi_dwell_times']
    poi_dwell_time_corrections = poi_dwell_times / (poi_dwell_times + 60)
    poi_categories = stuff['poi_categories']
    pretty_sub_category = np.array([SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x for x in poi_categories])
    if poi_time_counts is None:  # use saved poi_cbg_visits_list
        poi_cbg_visits_list = stuff['poi_cbg_visits_list'].copy()
        if end_idx is not None:
            poi_cbg_visits_list = poi_cbg_visits_list[:end_idx]
        if start_idx is not None:
            poi_cbg_visits_list = poi_cbg_visits_list[start_idx:]
        T = len(poi_cbg_visits_list)
        num_pois, num_cbgs = poi_cbg_visits_list[0].shape
    else:
        if end_idx is not None:
            poi_time_counts = poi_time_counts[:, :end_idx]
        if start_idx is not None:
            poi_time_counts = poi_time_counts[:, start_idx:]
        num_pois, T = poi_time_counts.shape
        num_cbgs = None
        poi_cbg_visits_list = None
        assert group_to_track is None  # can't differentiate CBGs if we only have overall POI visit counts
    if group_to_track is None:
        cbg_idx = None
    else:
        cbg_idx = stuff['cbg_idx_groups_to_track'][group_to_track]

    print('Aggregating data from %d hours' % T)
    total_weighted_transmission_rates = np.zeros(num_pois)
    total_weighted_occupancy = np.zeros(num_pois)
    total_weighted_people_per_sq_ft = np.zeros(num_pois)
    total_poi_visits = np.zeros(num_pois)  # denominator: only visits from CBGs of interest
    for t in range(T):
        if poi_time_counts is None:
            if cbg_idx is None:
                indicator = np.ones(num_cbgs)
            else:
                indicator = np.zeros(num_cbgs)
                indicator[cbg_idx] = 1.0
            poi_cbg_visits = poi_cbg_visits_list[t]
            poi_visits_from_any_cbg = poi_cbg_visits @ np.ones(num_cbgs)
            poi_visits_from_cbgs_of_interest = poi_cbg_visits @ indicator
        else:
            poi_visits_from_any_cbg = poi_time_counts[:, t]
            poi_visits_from_cbgs_of_interest = poi_visits_from_any_cbg

        time_spent = poi_dwell_times if overall_risk else poi_dwell_time_corrections
        if mode == 'normal':
            poi_transmission_rates = time_spent * poi_dwell_time_corrections * poi_visits_from_any_cbg / poi_areas
        elif mode == 'drop_area':
            poi_transmission_rates = time_spent * poi_dwell_time_corrections * poi_visits_from_any_cbg
        elif mode == 'drop_dwell_time':
            poi_transmission_rates = poi_visits_from_any_cbg / poi_areas
        elif mode == 'drop_visits':
            poi_transmission_rates = time_spent * poi_dwell_time_corrections / poi_areas
        elif mode == 'only_time_spent':
            poi_transmission_rates = time_spent
        else:  # mode is 'only_density'
            poi_transmission_rates = poi_dwell_time_corrections * poi_visits_from_any_cbg / poi_areas
        total_weighted_transmission_rates = total_weighted_transmission_rates + (poi_visits_from_cbgs_of_interest * poi_transmission_rates)
        total_weighted_occupancy = total_weighted_occupancy + (poi_visits_from_cbgs_of_interest * poi_visits_from_any_cbg)
        total_weighted_people_per_sq_ft = total_weighted_people_per_sq_ft + (poi_visits_from_cbgs_of_interest * poi_visits_from_any_cbg / poi_areas)
        total_poi_visits = total_poi_visits + poi_visits_from_cbgs_of_interest

    kept_idx = np.array(range(num_pois))[total_poi_visits >= 1]  # only want pois with at least 1 visits
    if verbose:
        print('Dropped %d/%d POIs with 0 visits in this time period.' % (num_pois - len(kept_idx), num_pois))
    poi_attributes_df = pd.DataFrame.from_dict({'poi_idx':kept_idx,
                                         'sub_category':poi_categories[kept_idx],
                                         'pretty_sub_category':pretty_sub_category[kept_idx],
                                         'total_num_visits':total_poi_visits[kept_idx],
                                         'area':poi_areas[kept_idx].copy(),
                                         'dwell_time':poi_dwell_times[kept_idx].copy(),
                                         'dwell_time_correction_factor':poi_dwell_time_corrections[kept_idx] ** 2,
                                         'avg_occupancy':total_weighted_occupancy[kept_idx] / total_poi_visits[kept_idx],
                                         'avg_people_per_sq_ft':total_weighted_people_per_sq_ft[kept_idx] / total_poi_visits[kept_idx],
                                         'avg_transmission_rate':total_weighted_transmission_rates[kept_idx] / total_poi_visits[kept_idx],
                                         })
    return poi_attributes_df

def get_category_attributes_from_poi_attributes(poi_attributes_df, categories, pop_size=None, verbose=True):
    category_num_visits = []
    category_num_pois = []
    category_avg_areas = []  # weighted average over POIs
    category_median_areas = []  # median (no weighting) over POIs
    category_avg_dwell_times = []
    category_median_dwell_times = []
    category_avg_dwell_time_correction_factors = []
    category_median_dwell_time_correction_factors = []
    category_avg_occupancies = []
    category_median_occupancies = []
    category_avg_densities = []
    category_median_densities = []
    category_avg_transmission_rates = []
    category_median_transmission_rates = []
    for cat in categories:
        subdf = poi_attributes_df[poi_attributes_df['pretty_sub_category'] == cat]
        if len(subdf) > 0:
            num_visits_per_poi = subdf['total_num_visits'].values
            category_num_visits.append(np.sum(num_visits_per_poi))
            category_num_pois.append(len(subdf))
            prop_visits_per_poi = num_visits_per_poi / np.sum(num_visits_per_poi)
            attributes_per_poi = subdf[['area', 'dwell_time', 'dwell_time_correction_factor', 'avg_occupancy',
                                        'avg_people_per_sq_ft', 'avg_transmission_rate']].values
            attribute_averages = (prop_visits_per_poi @ attributes_per_poi).T
            attribute_medians = np.median(attributes_per_poi, axis=0)  # median over POIs
        else:
            if verbose:
                print('Missing visits to any POI in %s' % cat)
            category_num_visits.append(0)
            category_num_pois.append(0)
            attribute_averages = np.ones(6) * np.nan
            attribute_medians = np.ones(6) * np.nan
        category_avg_areas.append(attribute_averages[0])
        category_median_areas.append(attribute_medians[0])
        category_avg_dwell_times.append(attribute_averages[1])
        category_median_dwell_times.append(attribute_medians[1])
        category_avg_dwell_time_correction_factors.append(attribute_averages[2])
        category_median_dwell_time_correction_factors.append(attribute_medians[2])
        category_avg_occupancies.append(attribute_averages[3])
        category_median_occupancies.append(attribute_medians[3])
        category_avg_densities.append(attribute_averages[4])
        category_median_densities.append(attribute_medians[4])
        category_avg_transmission_rates.append(attribute_averages[5])
        category_median_transmission_rates.append(attribute_medians[5])
    category_attributes_df = pd.DataFrame.from_dict({'category':categories,
                                                     'total_num_visits':category_num_visits,
                                                     'num_pois':category_num_pois,
                                                     'avg_area':category_avg_areas,
                                                     'median_area':category_median_areas,
                                                     'avg_dwell_time':category_avg_dwell_times,
                                                     'median_dwell_time':category_median_dwell_times,
                                                     'avg_dwell_time_correction_factor':category_avg_dwell_time_correction_factors,
                                                     'median_dwell_time_correction_factor':category_median_dwell_time_correction_factors,
                                                     'avg_occupancy':category_avg_occupancies,
                                                     'median_occupancy':category_median_occupancies,
                                                     'avg_people_per_sq_ft':category_avg_densities,
                                                     'median_people_per_sq_ft':category_median_densities,
                                                     'avg_transmission_rate':category_avg_transmission_rates,
                                                     'median_transmission_rate':category_median_transmission_rates})
    if pop_size is not None:
        category_attributes_df['num_visits_per_capita'] = category_attributes_df.total_num_visits.values / pop_size
    return category_attributes_df

def get_frac_infections_at_each_category_for_groups(timestring, groups, categories_to_plot, poi_categories):
    model, _, _, _, fast_to_load_results = load_model_and_data_from_timestring(timestring,
                                                                               load_fast_results_only=False,
                                                                               load_full_model=True)
    assert all([group in model.history for group in groups])
    group2fracs = {group:[] for group in groups}
    for group in groups:
        assert 'num_cases_per_poi' in model.history[group]  # doesn't always exist, we specify during experiment which
        # groups we want to track num_cases_per_poi for, since tracking this is expensive in terms of space
        num_cases_per_poi = model.history[group]['num_cases_per_poi']
        n_seeds = num_cases_per_poi.shape[0]
        pop_size = model.history[group]['total_pop']
        for s in range(n_seeds):
            frac_pop_infected = []
            for cat in categories_to_plot:
                cat_idx = poi_categories == cat
                num_cases_at_cat = np.sum(num_cases_per_poi[s][cat_idx])
                frac_pop_infected.append(num_cases_at_cat / pop_size)
            group2fracs[group].append(frac_pop_infected)
    return group2fracs

def plot_frac_infected_per_category_for_multiple_models(results_df, poi_and_cbg_characteristics, msa_name, ax, 
                                                        categories_to_plot, sort_categories=False, 
                                                        comparison='income', return_data=False):
    assert comparison in {'income', 'race'}
    if comparison == 'income':
        group1 = LOWINCOME
        group2 = HIGHINCOME
        label1 = 'bottom income decile'
        label2 = 'top income decile'
    else:
        group1 = NONWHITE
        group2 = WHITE
        label1 = 'decile with lowest % white'
        label2 = 'decile with highest % white'
    msa_df = results_df[results_df['MSA_name'] == msa_name]
    poi_categories = poi_and_cbg_characteristics[msa_name]['poi_categories']
    pretty_names = np.array([SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x for x in poi_categories])
    bottom_decile_fracs = []  # num_models x num_categories
    top_decile_fracs = []  # num_models x num_categories
    for ts in msa_df.timestring.values:
        group2fracs = get_frac_infections_at_each_category_for_groups(ts, [group1, group2], categories_to_plot, pretty_names)
        bottom_decile_fracs.extend(group2fracs[group1])
        top_decile_fracs.extend(group2fracs[group2])

    print('Num params * seeds:', len(bottom_decile_fracs))
    bottom_decile_fracs = np.array(bottom_decile_fracs)
    bottom_decile_mean = INCIDENCE_POP * np.mean(bottom_decile_fracs, axis=0)
    bottom_decile_min = INCIDENCE_POP * np.percentile(bottom_decile_fracs, LOWER_PERCENTILE, axis=0)
    bottom_decile_max = INCIDENCE_POP * np.percentile(bottom_decile_fracs, UPPER_PERCENTILE, axis=0)
    top_decile_fracs = np.array(top_decile_fracs)
    top_decile_mean = INCIDENCE_POP * np.mean(top_decile_fracs, axis=0)
    top_decile_min = INCIDENCE_POP * np.percentile(top_decile_fracs, LOWER_PERCENTILE, axis=0)
    top_decile_max = INCIDENCE_POP * np.percentile(top_decile_fracs, UPPER_PERCENTILE, axis=0)

    y_pos = range(len(categories_to_plot))
    if sort_categories:
        sorted_idx = np.argsort(bottom_decile_mean)  # sort category plotting by their impact
    else:
        sorted_idx = range(len(categories_to_plot)-1, -1, -1)  # want to go backwards so most popular category is on top
    ax.plot(bottom_decile_mean[sorted_idx], y_pos, label='bottom income decile', color='darkorchid', linewidth=2)
    ax.fill_betweenx(y=y_pos, x1=bottom_decile_min[sorted_idx],
                     x2=bottom_decile_max[sorted_idx], color='darkorchid', alpha=.3)
    ax.plot(top_decile_mean[sorted_idx], y_pos, label='top income decile', color='darkgoldenrod', linewidth=2)
    ax.fill_betweenx(y=y_pos, x1=top_decile_min[sorted_idx],
                     x2=top_decile_max[sorted_idx], color='darkgoldenrod', alpha=.5)
    ax.set_yticks(y_pos)
    labels = [categories_to_plot[i] for i in sorted_idx]
    ax.set_yticklabels(labels, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('Cumulative infections (per 100k)', fontsize=18)
    curr_lim = ax.get_xlim()
    ax.set_xlim(0, curr_lim[1])
    ax.grid(alpha=.3)
    if return_data:
        sorted_idx = np.argsort(-1 * bottom_decile_mean)  # most impactful to least impactful
        labels = [categories_to_plot[i] for i in sorted_idx]
        return (labels, bottom_decile_mean[sorted_idx], bottom_decile_min[sorted_idx], bottom_decile_max[sorted_idx], 
                top_decile_mean[sorted_idx], top_decile_min[sorted_idx], top_decile_max[sorted_idx])
    return labels
    
def make_mobility_comparison_line_plot(poi_and_cbg_characteristics,
                                       msa_name, min_datetime, max_datetime,
                                       group1, group1_label, group1_color,
                                       group2, group2_label, group2_color,
                                       ax, num_hours_to_agg=24, set_labels=True):
    stuff = poi_and_cbg_characteristics[msa_name]
    cbg_idx_groups_to_track, cbg_sizes = stuff['cbg_idx_groups_to_track'], stuff['cbg_sizes']
    poi_cbg_visits_list = stuff['poi_cbg_visits_list']
    hours = helper.list_hours_in_range(min_datetime, max_datetime)
    assert (len(hours) % num_hours_to_agg) == 0
    hours_to_plot = [hours[t] for t in np.arange(0, len(hours), num_hours_to_agg)]
    num_pois, num_cbgs = poi_cbg_visits_list[0].shape
    indicator_1 = np.zeros(num_cbgs)
    indicator_1[cbg_idx_groups_to_track[group1]] = 1.0
    pop_size_1 = np.sum(cbg_sizes[cbg_idx_groups_to_track[group1]])
    indicator_2 = np.zeros(num_cbgs)
    indicator_2[cbg_idx_groups_to_track[group2]] = 1.0
    pop_size_2 = np.sum(cbg_sizes[cbg_idx_groups_to_track[group2]])
    Y1 = []
    Y2 = []
    for t in range(len(hours_to_plot)):
        total_1 = 0
        total_2 = 0
        start_hr = t*num_hours_to_agg
        for hr in range(start_hr, start_hr+num_hours_to_agg):
            total_1 += np.sum(poi_cbg_visits_list[hr] @ indicator_1)
            total_2 += np.sum(poi_cbg_visits_list[hr] @ indicator_2)
        Y1.append(total_1 / pop_size_1)  # per-capita num visits over time period
        Y2.append(total_2 / pop_size_2)
    ax.plot_date(hours_to_plot, Y1, linestyle='-', marker='.', linewidth=2, label=group1_label, color=group1_color)
    ax.plot_date(hours_to_plot, Y2, linestyle='-', marker='.', linewidth=2, label=group2_label, color=group2_color)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.SU, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.2)

    if set_labels:
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Per capita mobility', fontsize=18)
        ax.set_title('%s MSA' % MSAS_TO_PRETTY_NAMES[msa_name], fontsize=20)
        ax.legend(fontsize=16)
    return hours_to_plot, Y1, Y2

def make_category_comparison_scatter_plot(attributes_df_1, attributes_df_2,
                                          attribute_to_plot, ax, color1, color2, title,
                                          xlabel, ylabel, x_lim=None, y_lim=None,
                                          plot_log=False, psi=1):

    categories = attributes_df_1.category.values
    vals_1 = attributes_df_1[attribute_to_plot].values
    visits_1 = attributes_df_1['num_visits_per_capita'].values
    vals_2 = attributes_df_2[attribute_to_plot].values
    visits_2 = attributes_df_2['num_visits_per_capita'].values
    avg_visits = (visits_1 + visits_2) / 2
    cats_and_visits = list(zip(categories, avg_visits))
    print('Top 10 categories with most visits:', sorted(cats_and_visits, key=lambda x:x[1], reverse=True)[:10])

    X = vals_1
    Y = vals_2
    if 'transmission_rate' in attribute_to_plot:
        X = X * psi
        Y = Y * psi
    sizes = 200 * avg_visits
    if plot_log:
        X = np.log(X)
        Y = np.log(Y)
        xlabel += ' (log)'
        ylabel += ' (log)'
    colors = []
    for x_pt, y_pt in zip(X, Y):
        if x_pt > y_pt:
            colors.append(color1)
        else:
            colors.append(color2)
    ax.scatter(X, Y, s=sizes, alpha=0.6, c=colors)

    max_val = max(max(X), max(Y))  # set x and y axes to have the same tick ranges
    offset = .1 * max_val
    if x_lim is None:
        ax.set_xlim(0 - offset, max_val + offset)
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is None:
        ax.set_ylim(0 - offset, max_val + offset)
    else:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.plot(list(ax.get_xlim()), list(ax.get_xlim()), color='grey', alpha=0.5, label='y=x')

    ax.tick_params(labelsize=16)
    ax.grid(alpha=.2)
    ax.legend(fontsize=16, loc='lower right')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    return X, Y, categories

def get_attribute_ratios_for_all_msas(group1, group2, poi_and_cbg_characteristics, categories_to_plot):
    attributes = ['avg_dwell_time', 'avg_people_per_sq_ft', 'avg_transmission_rate']
    col_names = ['%s_ratio' % a for a in attributes]
    all_results = []
    for msa_name in MSAS:
        print('Getting results for', msa_name)
        poi_attr_1 = get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, group_to_track=group1)
        cat_attr_1 = get_category_attributes_from_poi_attributes(poi_attr_1, categories_to_plot)
        poi_attr_2 = get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, group_to_track=group2)
        cat_attr_2 = get_category_attributes_from_poi_attributes(poi_attr_2, categories_to_plot)
        df = pd.DataFrame(cat_attr_1[attributes].values / cat_attr_2[attributes].values, columns=col_names)
        df['category'] = categories_to_plot
        df['MSA_name'] = msa_name
        all_results.append(df)
    all_results = pd.concat(all_results)
    return all_results

def plot_per_capita_category_visits(poi_and_cbg_characteristics, msa_name,
                                    group1, group1_label, group1_color,
                                    group2, group2_label, group2_color,
                                    categories_to_plot, ax, set_axis_labels=True):
    cbg_sizes = poi_and_cbg_characteristics[msa_name]['cbg_sizes']
    cbg_idx_groups_to_track = poi_and_cbg_characteristics[msa_name]['cbg_idx_groups_to_track']
    poi_attr_1 = get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, group_to_track=group1)
    group1_pop_size = np.sum(cbg_sizes[cbg_idx_groups_to_track[group1]])
    cat_attr_1 = get_category_attributes_from_poi_attributes(poi_attr_1, categories_to_plot, pop_size=group1_pop_size)
    X1 = cat_attr_1.num_visits_per_capita.values

    poi_attr_2 = get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, group_to_track=group2)
    group2_pop_size = np.sum(cbg_sizes[cbg_idx_groups_to_track[group2]])
    cat_attr_2 = get_category_attributes_from_poi_attributes(poi_attr_2, categories_to_plot, pop_size=group2_pop_size)
    X2 = cat_attr_2.num_visits_per_capita.values

    bar_pos = np.arange(0, 3*len(categories_to_plot), 3)
    ax.barh(bar_pos, X1, align='center', label=group1_label, color=group1_color)
    ax.barh(bar_pos-1, X2, align='center', label=group2_label, color=group2_color)
    ax.set_yticks(bar_pos-.5)
    ax.set_yticklabels(categories_to_plot, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    if set_axis_labels:
        ax.set_xlabel('Per capita visits to category', fontsize=18)
        ax.legend(fontsize=16, loc='lower right')
        ax.set_title(MSAS_TO_PRETTY_NAMES[msa_name], fontsize=20)

###################################################################################
# Functions for supplementary analyses
###################################################################################
def plot_hourly_poi_visits(mdl, ax, timesteps_to_plot=None, plot_daily_mean=True):
    total_poi_visits_by_timestep = mdl.POI_TIME_COUNTS.sum(axis=0)
    time_in_days = np.arange(len(total_poi_visits_by_timestep))/24. # time in days.
    hourly_alpha = 1 if not plot_daily_mean else 0.2
    hourly_color = 'blue' if not plot_daily_mean else 'grey'

    if timesteps_to_plot is not None:
        ax.plot(time_in_days[:timesteps_to_plot], total_poi_visits_by_timestep[:timesteps_to_plot], label='Hourly visits', alpha=hourly_alpha)
    else:
        ax.plot(time_in_days, total_poi_visits_by_timestep, label='Hourly visits', alpha=hourly_alpha)
    ax.set_xticks(range(0, math.ceil(max(time_in_days)) + 1, 7))
    ax.set_xlim([0, math.ceil(max(time_in_days))])
    ax.set_ylim([0, max(total_poi_visits_by_timestep)])
    ax.set_ylabel("Total visits to POIs", fontsize=16)
    ax.set_xlabel("Time in days", fontsize=16)

    if plot_daily_mean:
        smoothed_average = apply_smoothing(total_poi_visits_by_timestep, before=12, after=12)
        days_to_plot = time_in_days
        if timesteps_to_plot is not None:
            smoothed_average = smoothed_average[:timesteps_to_plot]
            days_to_plot = days_to_plot[:timesteps_to_plot]
        # don't plot very beginning or very end of the smoothed timeseries because smoothing can be misleading. 
        start_idx = 12
        end_idx = len(total_poi_visits_by_timestep) - 12
        smoothed_average = smoothed_average[start_idx:end_idx]
        days_to_plot = days_to_plot[start_idx:end_idx]

        ax.plot(days_to_plot, smoothed_average, label='Smoothed average', color='black', linewidth=4)
        ax.legend(fontsize=16)

    ax.grid(alpha=.5)

def plot_disease_spread_over_time(
    model,
    data_and_model_kwargs,
    cbg_mapper=None,
    path_to_save=None,
    timesteps_to_plot=None):

    # Extract mapping data.
    msa_name = data_and_model_kwargs['data_kwargs']['MSA_name']
    if cbg_mapper is None:
        cbg_mapper = helper.CensusBlockGroups(base_directory=PATH_FOR_CBG_MAPPER_BY_STATE,
                                             gdb_files=MSAS_TO_STATE_CBG_FILES[msa_name])
    msa_shapefile = gpd.read_file(PATH_TO_MSA_SHAPEFILES).to_crs(WGS_84_CRS)
    msa_shapefile['name_without_spaces'] = msa_shapefile['NAME'].map(
            lambda x:re.sub('[^0-9a-zA-Z]+', '_', x))
    msa_boundary = msa_shapefile.loc[msa_shapefile['name_without_spaces'] == msa_name]
    assert len(msa_boundary) == 1

    # Extract final fraction in LIR by CBG.
    all_log10_lir_by_cbg = {}
    for t in timesteps_to_plot:
        lir =  (np.array(model.full_history_for_all_CBGs['latent'][t]) + 
                np.array(model.full_history_for_all_CBGs['infected'][t]) + 
                np.array(model.full_history_for_all_CBGs['removed'][t]))
        lir_per_capita = lir / model.CBG_SIZES
        log10_lir_per_capita = np.log10(lir_per_capita + 1e-8)
        all_log10_lir_by_cbg[t] = log10_lir_per_capita

    min_val = np.percentile(all_log10_lir_by_cbg[min(timesteps_to_plot)], 10)
    max_val = np.percentile(all_log10_lir_by_cbg[max(timesteps_to_plot)], 90)



    for t in timesteps_to_plot:
        print("Plotting timestep %i" % t)
        lir_df = pd.DataFrame({'cbg':model.ALL_UNIQUE_CBGS,
                               'log10_lir_per_capita':all_log10_lir_by_cbg[t]})

        # match to geometry D.
        geometry_d = cbg_mapper.geometry_d.copy()
        geometry_d['cbg'] = geometry_d['GEOID_Data'].map(lambda x:x.split("US")[1]).astype(int)
        geometry_d = pd.merge(geometry_d,
                              lir_df,
                              on='cbg',
                              how='inner',
                              validate='one_to_one')

        print("Matched %i CBGs out of %i in SafeGraph data" % (len(geometry_d), len(lir_df)))

        geometry_d = geometry_d.dropna(subset=['log10_lir_per_capita'])
        print("After dropping NAs, %i rows" % len(geometry_d))


        # Filter for CBGs in MSA (just for plotting).
        nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(msa_name)
        msa_cbgs = set(msa_cbgs)
        geometry_d = geometry_d.loc[geometry_d['cbg'].map(lambda x:x in msa_cbgs)]
        print("After filtering for CBGs in MSA, %i CBGs" % (len(geometry_d)))

        fig = plt.figure(figsize=[18, 6])

        # make three plots. First is map. 
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Log10 fraction ever infected\nDay %i" % (t // 24), fontsize=16)
        geometry_d.plot(
                column=geometry_d['log10_lir_per_capita'].values,
                cmap='Spectral_r',
                linewidth=0.8,
                ax=ax,
                vmin=min_val,
                vmax=max_val,
                legend=True)
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

        # SLIR frac
        ax = fig.add_subplot(1, 3, 2)
        plot_slir_over_time(model, ax, timesteps_to_plot=t)

        ax = fig.add_subplot(1, 3, 3)
        plot_hourly_poi_visits(model, ax, timesteps_to_plot=t)

        plt.subplots_adjust(wspace=.3)


        if path_to_save is not None:
            fig.savefig(os.path.join(path_to_save % t), dpi=150, bbox_inches='tight')
    return cbg_mapper

def make_transmission_rate_sensitivity_plot(poi_and_cbg_characteristics, msa_name, 
                                            category_scoring, valid_categories):
    score_to_cats = {}
    for cat in valid_categories:
        score = category_scoring[cat]
        if score > 0:
            if score not in score_to_cats:
                score_to_cats[score] = []
            score_to_cats[score].append(cat)
    scores = sorted(score_to_cats.keys())
    print('Found categories in scores', scores)
    
    cats_to_annotate = []  # which categories to annotate in scatter plot
    for score, cats in score_to_cats.items():
        if len(cats) <= 3: 
            cats_to_annotate.extend(cats)
        else:  # if there are > 3 cats with this score, annotate 3 most popular 
            # (assume valid_categories in order of popularity)
            top3_cats = [cat for cat in valid_categories if category_scoring[cat] == score][:3]
            cats_to_annotate.extend(top3_cats)
    print('Will annotate:', cats_to_annotate)
    
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    fig.suptitle('%s: first week of March 2020' % MSAS_TO_PRETTY_NAMES[msa_name], fontsize=20)
    plt.subplots_adjust(wspace=.3, top=0.90)
    modes = ['normal', 'only_time_spent', 'drop_dwell_time']
    labels = ['Original parametric equation', 'Only time spent', 'Only density']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    all_results = []
    for i in range(len(modes)):
        mode, label, color, ax = modes[i], labels[i], colors[i], axes[i]
        poi_attr = get_poi_attributes_for_msa(poi_and_cbg_characteristics, msa_name, 
                                              start_idx=0, end_idx=168, mode=modes[i])
        y_min = 100
        y_max = -100
        kept_scores = []
        medians = []
        all_cats = []
        all_y = []
        for score in scores:
            cat_attr = get_category_attributes_from_poi_attributes(poi_attr, score_to_cats[score])
            cat_attr = cat_attr[cat_attr['total_num_visits'] > 0]
            n_cats = len(cat_attr)
            if n_cats > 0:
                x = np.ones(n_cats) * score
                y = cat_attr.avg_transmission_rate.values
                ax.scatter(x, y, alpha=0.8, color=color)
                kept_scores.append(score)
                medians.append(np.median(y))
                all_cats.extend(cat_attr.category.values)
                all_y.extend(y)
                y_min = min(y_min, min(y))
                y_max = max(y_max, max(y))
                for cat, x_pt, y_pt in zip(cat_attr.category.values, x, y):
                    if cat in cats_to_annotate:
                        if score != max(scores):
                            ax.annotate(cat, (x_pt, y_pt), ha='left', fontsize=14)
                        else:
                            ax.annotate(cat, (x_pt, y_pt), ha='right', fontsize=14)

        ax.plot(kept_scores, medians, color='grey', label='median transmission\nrate', alpha=0.5)
        ax.legend(fontsize=14, loc='lower right')
        ax.set_xticks(kept_scores)
        ax.set_xlabel('External risk score', fontsize=16)
        if i % 3 == 0:
            ax.set_ylabel('Avg transmission rate', fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title(label, fontsize=18)
        offset = (y_max - y_min) * .1
        ax.set_ylim(y_min - offset, y_max + offset)
        results_df = pd.DataFrame({'category':all_cats, 'avg_transmission_rate':all_y})
        results_df['mode'] = mode
        all_results.append(results_df)
    all_results = pd.concat(all_results)
    return all_results

from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap

def make_identifiability_plot(fig, all_results_for_identifiability_plot, make_contour_plot,
                              key_to_sort_by, metric, 
                              min_threshold=None, do_logsumexp=False, plot_ratio=False):
    
    cmap = plt.cm.get_cmap('cool')
    cmap_colors = cmap(np.arange(cmap.N))    

    N = 256
    spacing = np.concatenate((
        np.linspace(0, 0.33, N/2, endpoint=False),
        np.linspace(0.33, 1.0, N/2)))
    colors = cmap_colors[:, :3]
    spacing_and_colors = []
    for i in range(N):
        spacing_and_colors.append((spacing[i], colors[i, :]))

    cmap = LinearSegmentedColormap.from_list(
        name='new_cool',
        colors=spacing_and_colors)
    
    param1 = 'home_beta'
    param2 = 'poi_psi'
    pretty_param_names = {
        "home_beta":r"$\beta$ (base CBG transmission rate)", 
        'poi_psi':'$\Psi$ (scaling for POI transmission)'}

    for subplot_idx, MSA in enumerate(sorted(list(MSAS_TO_PRETTY_NAMES.keys()))):
        print(MSA)
        # load all models for MSA
        all_results = all_results_for_identifiability_plot[MSA].copy()

        all_results = all_results.loc[(all_results['poi_psi'] > 0)]

        # find best-fit model, using our normal metric. 
        all_results = all_results.sort_values(by=key_to_sort_by)
        assert all_results['poi_psi'].iloc[0] > 0
        best_fit_cumulative_predicted_cases = all_results.iloc[0]['loss_dict_cumulative_predicted_cases']
        best_fit_params = (all_results.iloc[0][param1], all_results.iloc[0][param2])        
        n_seeds = best_fit_cumulative_predicted_cases.shape[0]
        seed_cutoff = n_seeds // 2
        assert seed_cutoff * 2 == n_seeds

        # compute the "ground truth": mean of first half of seeds for best-fit model (cumulative predictions)
        y_true_cumulative = best_fit_cumulative_predicted_cases[:seed_cutoff, :].mean(axis=0)
        y_true_cumulative = np.round(y_true_cumulative, 0).astype(int)  # y_true needs to be int for poisson
        losses = []
        for i in range(len(all_results)):
            # prediction - use second half of seeds. 
            y_pred_cumulative = all_results.iloc[i]['loss_dict_cumulative_predicted_cases'][seed_cutoff:, :]
            # loss is daily RMSE. 
            losses.append(compute_loss(y_true=y_true_cumulative, 
                                       y_pred=y_pred_cumulative,
                                       metric=metric,
                                       min_threshold=min_threshold,
                                       compare_daily_not_cumulative=True,
                                       do_logsumexp=do_logsumexp))
        print("Indices of models with five lowest losses: %s. Min loss = %.4f." % 
             (np.argsort(losses)[:5], min(losses)))

        if make_contour_plot:

            df_to_plot = all_results[['home_beta', 'poi_psi', 'p_sick_at_t0', key_to_sort_by]].copy()
            df_to_plot['simulated_loss'] = losses
            assert df_to_plot[key_to_sort_by].iloc[0] == df_to_plot[key_to_sort_by].values.min()
            grouped = df_to_plot.groupby([param1, param2])[key_to_sort_by].min().reset_index()
            M = len(set(grouped[param1]))
            N = len(set(grouped[param2]))

            X = np.reshape(grouped[param1].values, [M, N])
            Y = np.reshape(grouped[param2].values, [M, N])
            Z = np.reshape(grouped[key_to_sort_by].values, [M, N])
            if plot_ratio:
                Z = np.log10(Z/Z.min())
            else:
                offset = Z.min()
                log_offset = np.log10(offset)
                Z = np.log10(Z) - log_offset

            upper_boundary = 5
            Z[Z > np.log10(upper_boundary)] = np.log10(upper_boundary) # clip so colormap doesn't turn it white.             
            ax = fig.add_subplot(5, 2, subplot_idx + 1)
            ax.contourf(X, Y, Z, levels=list(np.linspace(0, np.log10(upper_boundary), 20)), cmap=cmap)#, vmin=0, vmax=np.log10(upper_boundary))#,norm=matplotlib.colors.LogNorm(vmin=1, vmax=1000, bins=50))
            ax.set_title(MSAS_TO_PRETTY_NAMES[MSA]);
            if subplot_idx >= 8:
                ax.set_xlabel(pretty_param_names[param1])
            if subplot_idx % 2 == 0:
                ax.set_ylabel(pretty_param_names[param2])
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(Z)
            m.set_clim(0., np.log10(upper_boundary))
            cb = plt.colorbar(m, ticks=[0, np.log10(upper_boundary)], extend='max')#, boundaries=np.linspace(0, 2, 10))
            if plot_ratio:                
                cb.ax.set_yticklabels(['1', str(upper_boundary)])
                cb.ax.set_ylabel("Ratio of %s" % metric)
            else:
                cb.ax.set_yticklabels([f'{offset:.0f}', f'{upper_boundary * offset:.0f}'])
                cb.ax.set_ylabel("%s" % metric)
                cb.ax.yaxis.set_label_coords(2.0, 0.5)

            points = grouped.loc[grouped[key_to_sort_by] <= (ACCEPTABLE_LOSS_TOLERANCE*grouped[key_to_sort_by].min()), 
                                         [param1, param2]].values
            print('Num best fit models:', len(points))
            try:
                hull = ConvexHull(points)                                     
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], color='white')
            except:
                x_pts = points[:, 0]
                y_pts = points[:, 1]
                ax.scatter(x_pts, y_pts, color='white')
        
        else:
            ax = fig.add_subplot(5, 2, subplot_idx + 1)
            ax.plot(range(1, len(losses) + 1), losses)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.xticks([1, 10, 100, 1000], ['Best-fit model', '10', '100', '1000'])
            ax.set_xlim([1, len(losses)])            
            if subplot_idx >= 8:
                ax.set_xlabel("Model rank by %s on real data" % metric)
            if subplot_idx % 2 == 0:
                ax.set_ylabel("%s on simulated data" % metric)
            ax.set_title(MSAS_TO_PRETTY_NAMES[MSA])
    plt.show()

def get_frac_infected_over_time_per_category(results_df, poi_and_cbg_characteristics, msa_name, categories_to_plot):
    msa_df = results_df[results_df['MSA_name'] == msa_name]
    poi_categories = poi_and_cbg_characteristics[msa_name]['poi_categories']
    pretty_names = np.array([SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x for x in poi_categories])
    results_per_seed = []
    for ts in msa_df.timestring.values:
        model, _, _, _, fast_to_load_results = load_model_and_data_from_timestring(ts,
                                                                               load_fast_results_only=False,
                                                                               load_full_model=True)
        num_cases_per_poi = model.history['all']['num_cases_per_poi']
        pop_size = model.history['all']['total_pop']
        assert len(num_cases_per_poi.shape) == 3  # must be seed x poi x time
        for s in range(len(num_cases_per_poi)):  # iterate through seeds
            total_frac_infected_at_pois_per_day = np.sum(num_cases_per_poi[s], axis=0) / pop_size  # sum over all pois
            frac_infected_per_cat_and_day = []
            for cat in categories_to_plot:
                cat_idx = pretty_names == cat
                assert np.sum(cat_idx) > 10  # there should be at least 10 POIs in this category
                frac_infected_at_cat_per_day = np.sum(num_cases_per_poi[s][cat_idx], axis=0) / pop_size  # sum over pois within cat
                frac_infected_per_cat_and_day.append(frac_infected_at_cat_per_day)
            results_per_seed.append((frac_infected_per_cat_and_day, total_frac_infected_at_pois_per_day))
    print('Num params * seeds:', len(results_per_seed))
    return results_per_seed
    
def plot_stacked_infection_proportions_over_categories(results_per_seed, categories_in_results, 
                                                       categories_to_plot, msa_name, ax, y_max=1.0, smooth=True):
    '''
    Note: relies on output from get_frac_infections_per_category_over_seeds.
    '''
    dates = helper.list_datetimes_in_range(MIN_DATETIME, MAX_DATETIME)
    bottom = np.zeros(len(dates))
    for cat in categories_to_plot:
        assert cat in categories_in_results
        i = categories_in_results.index(cat)
        prop_infections_over_seeds = []
        for s in range(len(results_per_seed)):
            frac_infected = results_per_seed[s][0][i]  # fraction of population infected at this cat per day
            total_frac_infected = results_per_seed[s][1]  # total fraction of population infected at POIs per day
            prop_infections = frac_infected / total_frac_infected  # proportion of POI infections at this cat per day
            prop_infections_over_seeds.append(prop_infections)
        mean_prop_infections_per_day = np.mean(np.array(prop_infections_over_seeds), axis=0)  # average over seeds
        if smooth:
            mean_prop_infections_per_day = apply_smoothing(mean_prop_infections_per_day, before=3, after=3)
        top = bottom + mean_prop_infections_per_day
        ax.fill_between(dates, bottom, top, label=cat, alpha=0.5)
        bottom = top
    ceiling = np.ones(len(bottom)) * y_max
    assert all(bottom < ceiling)
    ax.fill_between(dates, bottom, ceiling, label='Other')
    
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.SU, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Date', fontsize=16)
    ylabel = 'Proportion of daily\nPOI infections'
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title(MSAS_TO_PRETTY_NAMES[msa_name], fontsize=20)