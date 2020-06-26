import numpy as np
import matplotlib.pyplot as plt
import time
import os
import datetime
from collections import Counter
import math
import pickle
import re
import sys
import psutil
import IPython
import seaborn as sns
from scipy import sparse

class Model:
    def __init__(self,
                 starting_seed=0,
                 num_seeds=1,
                 approx_method='poisson',
                 debug=False,
                 stochastic=True,
                 clip_poisson_approximation=True,
                 ipf_final_match='cbg',
                 ipf_num_iter=100):

        self.stochastic = stochastic
        self.starting_seed = starting_seed
        self.num_seeds = num_seeds
        self.approx_method = approx_method
        assert approx_method in ['true', 'poisson', 'normal']
        if self.approx_method == 'normal':
            raise Exception("Currently normal and true currently don't implement base rate or random seeds properly - if you want to use these you have to fix that.")
            self.get_new_cases = self.get_new_cases_normal_approx
        elif self.approx_method == 'poisson':
            self.get_new_cases = self.get_new_cases_poisson_approx
        elif self.approx_method == 'true':
            raise Exception("Currently normal and true currently don't implement base rate or random seeds properly - if you want to use these you have to fix that.")
            self.get_new_cases = self.get_new_cases_binomial_true
        self.debug = debug
        self.exo_initialized = False
        self.endo_initialized = False
        self.ipf_final_match = ipf_final_match
        assert ipf_final_match in ['cbg', 'poi']
        self.ipf_num_iter = ipf_num_iter
        self.clip_poisson_approximation = clip_poisson_approximation

        # TODO: give each model its own np random generator
        np.random.seed(self.starting_seed)

    def init_exogenous_variables(self,
                                 poi_cbg_proportions,
                                 poi_time_counts,
                                 poi_areas,
                                 cbg_sizes,
                                 all_unique_cbgs,
                                 cbgs_to_idxs,
                                 p_sick_at_t0,
                                 all_hours,
                                 poi_cbg_visits_list=None,
                                 poi_dwell_time_correction_factors=None,
                                 all_states=None,
                                 cbg_densities=None,
                                 cbg_idx_groups_to_track=None,
                                 intervention_cost=None,
                                 cbg_idx_to_seed_in=None,
                                 cbg_day_prop_out=None,
                                 just_compute_r0=False,
                                 latency_period=96,  # 4 days
                                 infectious_period=84,  # 3.5 days
                                 confirmation_rate=.1,
                                 confirmation_lag=168,  # 7 days
                                 death_rate=.0066,
                                 death_lag=432,  # 18 days
                                 poi_psi=None, #poi_psi=500,
                                 home_beta=None, #home_beta=.01,
                                 home_psi=None,
                                 poi_subcategory_types=None): #home_psi=1e-5):

        self.M = len(poi_areas)
        self.N = len(cbg_sizes)
        self.T = len(all_hours)
        # POI variables
        self.POI_TIME_COUNTS = poi_time_counts  # POI x hour
        self._build_proportions_matrix(poi_cbg_proportions)
        self.PSI = poi_psi
        self.POI_AREAS = poi_areas
        self.DWELL_TIME_CORRECTION_FACTORS = poi_dwell_time_correction_factors
        self.POI_FACTORS = self.PSI / poi_areas
        self.POI_SUBCATEGORY_TYPES = poi_subcategory_types
        if poi_dwell_time_correction_factors is not None:
            self.POI_FACTORS = poi_dwell_time_correction_factors * self.POI_FACTORS
            print('Adjusted POI transmission rates with dwell time correction factors')
            self.included_dwell_time_correction_factors = True
        else:
            self.included_dwell_time_correction_factors = False
        self.POI_CBG_VISITS_LIST = poi_cbg_visits_list
        if self.POI_CBG_VISITS_LIST is not None:
            print('Received POI_CBG_VISITS_LIST, will NOT be computing hourly matrices on the fly')
            assert len(self.POI_CBG_VISITS_LIST) == self.T
            assert self.POI_CBG_VISITS_LIST[0].shape == (self.M, self.N)
            assert cbg_day_prop_out is None
        else:
            # will use this matrix to compute hourly counts; must match all_hours length
            assert self.POI_TIME_COUNTS.shape[1] == self.T

        # CBG variables
        self.CBG_SIZES = cbg_sizes  # has dimensions CBG, population size of each CBG
        self.cbg_day_prop_out = cbg_day_prop_out  # CBG x day
        if self.cbg_day_prop_out is not None:
            assert(self.N == self.cbg_day_prop_out.shape[0])
            # number of days = number of hours x 24
            assert(int(self.T / 24) == self.cbg_day_prop_out.shape[1])
        if cbg_densities is None: # assume constant transmission rate, irrespective of how many people per square mile are in CBG.
            assert home_beta is not None
            assert home_psi is None
            self.HOME_BETA = home_beta
            self.density_based_home_rates = False
        else: # assume transmission rate depends on how many people live in CBG.
            assert home_beta is None
            assert home_psi is not None
            self.HOME_BETA = home_psi * cbg_densities
            self.density_based_home_rates = True
        self.cbg_idx_groups_to_track = cbg_idx_groups_to_track if cbg_idx_groups_to_track is not None else {}
        assert ('all' not in self.cbg_idx_groups_to_track)
        self.cbg_idx_groups_to_track['all'] = np.arange(self.N)
        self.cbg_idx_to_seed_in = cbg_idx_to_seed_in  # which CBGs start off with sick patients
        self.ALL_UNIQUE_CBGS = all_unique_cbgs  # list of CBG names
        self.CBGS_TO_IDXS = cbgs_to_idxs  # mapping of CBG name to index

        self.LATENCY_PERIOD = latency_period
        self.INFECTIOUS_PERIOD = infectious_period
        self.ALL_STATES = all_states
        self.all_hours = all_hours
        self.P_SICK_AT_T0 = p_sick_at_t0  # percentage of CBG that starts off sick
        self.just_compute_r0 = just_compute_r0
        self.INTERVENTION_COST = intervention_cost

        self.confirmation_rate = confirmation_rate
        self.confirmation_lag = confirmation_lag
        self.death_rate = death_rate
        self.death_lag = death_lag

        array_params = [self.POI_FACTORS, self.CBG_SIZES, self.POI_TIME_COUNTS]
        number_params = [self.LATENCY_PERIOD, self.INFECTIOUS_PERIOD]
        if self.density_based_home_rates:
            array_params.append(self.HOME_BETA)
        else:
            number_params.append(self.HOME_BETA)
        for arr in array_params:
            assert (arr >= 0).all()
        for number in number_params:
            assert number >= 0
        self.exo_initialized = True

    def _build_proportions_matrix(self, proportion_dicts):
        '''
        Convert the list of dictionaries, each mapping CBG to proportion
        for that POI, to a POI x CBG matrix.
        '''
        assert(self.M is not None and self.N is not None)
        # static, historical CBG proportions
        assert len(proportion_dicts) == self.M

        self.POI_CBG_PROPORTIONS = np.zeros((self.M, self.N))
        for poi, cbg_dict in enumerate(proportion_dicts):
            for cbg, prop in cbg_dict.items():
                self.POI_CBG_PROPORTIONS[poi, cbg] = prop
        assert (self.POI_CBG_PROPORTIONS >= 0).all()
        assert (self.POI_CBG_PROPORTIONS.sum(axis=1) <= 1 + 1e-6).all()
        if self.M > 1000: # only check this if we're not just testing.
            assert (self.POI_CBG_PROPORTIONS.sum(axis=1) < .5).mean() < 0.01 # make sure not very many POIs seem to have lots of people who are coming from outside of the CBG set.

    def init_endogenous_variables(self):
        assert(self.exo_initialized)
        # create sick people
        self.P0 = np.random.binomial(
            self.CBG_SIZES,
            self.P_SICK_AT_T0,
            size=(self.num_seeds, self.N))
        if self.cbg_idx_to_seed_in is not None:
            multiplier = np.zeros(self.N)
            multiplier[self.cbg_idx_to_seed_in] = 1.
            self.P0 = self.P0 * multiplier

        self.cbg_latent = self.P0
        self.cbg_infected = np.zeros((self.num_seeds, self.N))
        self.cbg_removed = np.zeros((self.num_seeds, self.N))
        self.cases_to_confirm = np.zeros((self.num_seeds, self.N))
        self.new_confirmed_cases = np.zeros((self.num_seeds, self.N))
        self.deaths_to_happen = np.zeros((self.num_seeds, self.N))
        self.new_deaths = np.zeros((self.num_seeds, self.N))

        # Monitor clipping of Poisson approximation.
        self.clipping_monitor = {
        'num_base_infection_rates_clipped':[],
        'num_active_pois':[],
        'num_poi_infection_rates_clipped':[],
        'num_cbgs_active_at_pois':[],
        'num_cbgs_with_clipped_poi_cases':[]}

        # Keep track of how many people are in SLIR at each timestep for various groups [low SES etc] as well as whole population.
        self.history = {}
        for group in self.cbg_idx_groups_to_track:
            group_idxs = self.cbg_idx_groups_to_track[group]
            self.history[group] = {}
            self.history[group]['total_pop'] = np.sum(self.CBG_SIZES[group_idxs])
            self.history[group]['num_cbgs'] = len(group_idxs)
            for compartment in [
                'new_cases',
                'new_cases_from_poi',
                'new_cases_from_base',
                'new_confirmed_cases',
                'new_deaths',
                'susceptible',
                'latent',
                'infected',
                'removed',
            ]:
                self.history[group][compartment] = np.zeros((self.num_seeds, self.T))  # differs across seeds
            for compartment in [
                'num_out',
                'num_cbgs_with_no_out'
            ]:
                self.history[group][compartment] = np.zeros(self.T)  # non-stochastic
            for compartment in [
                'ipf_iter_col_num_out',
                'ipf_iter_row_num_out'
            ]:
                self.history[group][compartment] = np.zeros(self.ipf_num_iter)
            self.history[group]['before_ipf_num_out'] = 0

        # dynamic, CBG proportions for current day
        self.POI_CBG_PROPORTIONS = sparse.csr_matrix(self.POI_CBG_PROPORTIONS)
        self.poi_sum_proportions = np.sum(self.POI_CBG_PROPORTIONS, axis=1)
        self.poi_cbg_visit_history = []  # to store ipf output
        self.estimated_R0 = None
        self.endo_initialized = True

    def simulate_disease_spread(self, verbosity=24,
                                simulate_cases=False, simulate_deaths=False,
                                groups_to_track_num_cases_per_poi=None):
        '''
        Simulate disease spread over the bipartite network.
        T: the total number of iterations to run; however, if the disease
           dies out before T, the simulation stops
        verbosity: how often to print output
        '''
        assert(self.exo_initialized and self.endo_initialized)
        self.simulate_cases = simulate_cases
        if not self.simulate_cases:
            for group in self.cbg_idx_groups_to_track:
                del self.history[group]['new_confirmed_cases']
        self.simulate_deaths = simulate_deaths
        if not self.simulate_deaths:
            for group in self.cbg_idx_groups_to_track:
                del self.history[group]['new_deaths']

        if groups_to_track_num_cases_per_poi is None:
            groups_to_track_num_cases_per_poi = {}
        self.groups_to_track_num_cases_per_poi = groups_to_track_num_cases_per_poi
        for group in self.groups_to_track_num_cases_per_poi:
            print('Tracking num cases per poi for %s' % group)
            self.history[group]['num_cases_per_poi'] = np.zeros((self.num_seeds, self.M))

        if verbosity > 0:
            print('=== PARAMETERS ===')
            if self.density_based_home_rates:
                print('poi_psi = %s, p_sick_at_t0 = %s, num_hours = %d' % (
                    self.PSI, self.P_SICK_AT_T0, self.T))
                print('CBG transmission rates for first 10 CBGs')
                print(self.HOME_BETA[:10])
            else:
                # print all together
                print('poi_psi = %s, home_beta = %s, p_sick_at_t0 = %s, num_hours = %d' % (
                    self.PSI, self.HOME_BETA, self.P_SICK_AT_T0, self.T))
            if self.included_dwell_time_correction_factors:
                eq = 'psi * dwell_time_factor / area'
            else:
                eq = 'psi / area'
            print('POI factors (%s) for first 10 POIs' % eq)
            print(self.POI_FACTORS[:10])
            print('simulating confirmed cases: %s, simulating deaths: %s' % (
                self.simulate_cases, self.simulate_deaths))
            print(f'=== RESULTS ({self.num_seeds} seeds) ===')
            start_time = time.time()

        t = 0
        while t < self.T:
            iter_t0 = time.time()
            if (verbosity > 0) and (t % verbosity == 0):
                L = np.sum(self.cbg_latent, axis=1)
                I = np.sum(self.cbg_infected, axis=1)
                R = np.sum(self.cbg_removed, axis=1)
                print((
                    f't={t:3d}: L={np.mean(L):5.1f} ({np.std(L):5.1f})'
                    f'   I={np.mean(I):5.1f} ({np.std(I):5.1f})'
                    f'   R={np.mean(R):5.1f} ({np.std(R):5.1f})'))
            self.update_states(t)
            if self.debug and verbosity > 0 and t % verbosity == 0:
                print('Num active POIs: %d. Num with infection rates clipped: %d' % (self.num_active_pois, self.num_poi_infection_rates_clipped))
                print('Num CBGs active at POIs: %d. Num with clipped num cases from POIs: %d' % (self.num_cbgs_active_at_pois, self.num_cbgs_with_clipped_poi_cases))
            if self.debug:
                print("Time for iteration %i: %2.3f seconds" % (t, time.time() - iter_t0))

            if np.max(self.cbg_latent + self.cbg_infected) < 1:
                print('Disease died off after t=%d. Stopping experiment.' % t)
                if t < self.T-1:
                    # need to fill in trailing 0's in self.history
                    self.fill_remaining_history(t)
                break
            t += 1

        all_infected = self.cbg_latent + self.cbg_infected + self.cbg_removed
        if self.N <= 10:
            print('Final state after %d rounds: L+I+R=%s' % (t, self.format_floats(all_infected)))
        total = np.sum(all_infected, axis=1)
        print(f'Average number of people infected across random seeds: {np.mean(total):.3f}')

        if self.just_compute_r0:
            assert self.cbg_latent.sum() == 0
            assert self.cbg_infected.sum() == 0

            initial_cases = self.P0.sum(axis=1)
            self.estimated_R0 = {'R0':1.*(total - initial_cases) / initial_cases}
            assert self.estimated_R0['R0'].shape  == total.shape == initial_cases.shape
            print("Mean initial cases across seeds: %2.3f; new cases from initial: %2.3f; estimated R0: %2.3f" %
                (initial_cases.mean(), (total - initial_cases).mean(), self.estimated_R0['R0'].mean()))

            total_base = self.history['all']['new_cases_from_base'].sum(axis=1)
            total_poi = self.history['all']['new_cases_from_poi'].sum(axis=1)
            assert total_base.shape == total_poi.shape == initial_cases.shape
            self.estimated_R0['R0_base'] = 1.*total_base / initial_cases
            self.estimated_R0['R0_POI'] = 1.*total_poi / initial_cases
            assert np.allclose(self.estimated_R0['R0_base'] + self.estimated_R0['R0_POI'], self.estimated_R0['R0'])

        end_time = time.time()
        print('Simulation time = %.3fs -> %.3fs per iteration' %
            (end_time - start_time, (end_time - start_time)/t))

    def update_states(self, t):
        '''
        Applies one round of updates. First, we compute the infection rates
        at each POI depending on which CBGs are visiting it at time t. Based
        on the home and POI infection rates, we compute the number of new
        cases per CBG. Then, we update the SLIR states accordingly.
        '''
        self.get_new_cases(t)
        new_infectious = self.get_new_infectious()
        new_removed = self.get_new_removed()

        if not self.just_compute_r0:
            # normal case.
            self.cbg_latent = self.cbg_latent + self.cbg_new_cases - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed
        else:
            # if we want to calibrate R0, don't allow anyone new to become infected - just put new_cases in removed.
            self.cbg_latent = self.cbg_latent - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed + self.cbg_new_cases

        if self.simulate_cases and t > 0:  # 0 cases to confirm at time t
            self.new_confirmed_cases = np.random.binomial(self.cases_to_confirm.astype(int), 1/self.confirmation_lag)
            new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), self.confirmation_rate)
            self.cases_to_confirm = self.cases_to_confirm + new_cases_to_confirm - self.new_confirmed_cases
        if self.simulate_deaths and t > 0:
            self.new_deaths = np.random.binomial(self.deaths_to_happen.astype(int), 1/self.death_lag)
            new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), self.death_rate)
            self.deaths_to_happen = self.deaths_to_happen + new_deaths_to_happen - self.new_deaths
        self.update_history(t)

    def update_history(self, t):
        for group in self.cbg_idx_groups_to_track:
            group_idxs = self.cbg_idx_groups_to_track[group]
            self.history[group]['new_cases'][:, t] = np.sum(self.cbg_new_cases[:, group_idxs], axis=1)
            self.history[group]['new_cases_from_poi'][:, t] = np.sum(self.cbg_new_cases_from_poi[:, group_idxs], axis=1)
            self.history[group]['new_cases_from_base'][:, t] = np.sum(self.cbg_new_cases_from_base[:, group_idxs], axis=1)
            self.history[group]['latent'][:, t] = np.sum(self.cbg_latent[:, group_idxs], axis=1)
            self.history[group]['infected'][:, t] = np.sum(self.cbg_infected[:, group_idxs], axis=1)
            self.history[group]['removed'][:, t] = np.sum(self.cbg_removed[:, group_idxs], axis=1)
            self.history[group]['susceptible'][:, t] = (
                self.history[group]['total_pop']
                - self.history[group]['latent'][:, t]
                - self.history[group]['infected'][:, t]
                - self.history[group]['removed'][:, t])
            self.history[group]['num_out'][t] = np.sum(self.cbg_num_out[group_idxs])
            self.history[group]['num_cbgs_with_no_out'][t] = np.sum(self.cbg_num_out[group_idxs] <= 1e-6)
            if self.simulate_cases and t > 0:  # could not have confirmed cases at time 0 (no one in infectious yet)
                self.history[group]['new_confirmed_cases'][:, t] = np.sum(self.new_confirmed_cases[:, group_idxs], axis=1)
            if self.simulate_deaths and t > 0:  # could not have confirmed cases at time 0
                self.history[group]['new_deaths'][:, t] = np.sum(self.new_deaths[:, group_idxs], axis=1)
            if group in self.groups_to_track_num_cases_per_poi:
                group_indicator = np.zeros(self.N)
                group_indicator[group_idxs] = 1.0
                for s in range(self.num_seeds):
                    seed_poi_cbg_infected = self.cbg_num_cases_per_poi[s]
                    seed_poi_group_infected = seed_poi_cbg_infected @ group_indicator  # 1 x M
                    prev_total = self.history[group]['num_cases_per_poi'][s]
                    self.history[group]['num_cases_per_poi'][s] = prev_total + seed_poi_group_infected

    def fill_remaining_history(self, t):
        for group in self.cbg_idx_groups_to_track:
            for state in ['susceptible', 'latent', 'infected', 'removed']:
                final_values = self.history[group][state][:, t]
                # make sure we are not overwriting anything
                assert np.sum(self.history[group][state][:, t+1:]) < 1e-10
                remaining_t = self.T - t - 1
                self.history[group][state][:, t+1:] = np.broadcast_to(final_values, (remaining_t, self.num_seeds)).T

    def get_new_cases_poisson_approx(self, t):
        '''
        Determines the number of new cases per CBG. This depends on the CBG's
        home infection rate and the infection rates of the POIs that members
        from this CBG visited at time t. If the model is stochastic, the
        number of new cases is drawn randomly; otherwise, the expectation of the
        random variable is used.

        This method computes the weighted rates then uses a Poisson approximation
        to the binomial.
        '''
        # M is number of POIs
        # N is number of CBGs
        # S is number of seeds

        ### Compute CBG densities and infection rates
        # S x N
        cbg_densities = self.cbg_infected / self.CBG_SIZES  # infection density
        num_sus = np.clip(self.CBG_SIZES - self.cbg_latent - self.cbg_infected - self.cbg_removed, 0, None)
        sus_frac = num_sus / self.CBG_SIZES
        assert (cbg_densities >= 0).all()
        assert (cbg_densities <= 1).all()
        assert (sus_frac >= 0).all()
        assert (sus_frac <= 1).all()

        # S x N
        if self.PSI > 0:
            cbg_base_infection_rates = self.HOME_BETA * cbg_densities # Our model: can only be infected by people in your home CBG.
        else:
            cbg_base_infection_rates = self.HOME_BETA * np.ones(self.cbg_infected.shape) * (self.cbg_infected.sum() / self.CBG_SIZES.sum()) # ablation: standard model with uniform mixing.
        self.num_base_infection_rates_clipped = np.sum(cbg_base_infection_rates > 1)
        cbg_base_infection_rates = np.clip(cbg_base_infection_rates, None, 1.0)


        ### Load or compute POI x CBG matrix

        # Try to load POI x CBG matrix
        if self.POI_CBG_VISITS_LIST is not None:
            # M x N
            poi_cbg_visits = self.POI_CBG_VISITS_LIST[t]
            poi_visits = poi_cbg_visits @ np.ones(poi_cbg_visits.shape[1])  # faster than summing axis=1
        # Otherwise, compute it
        else:
            # M x 1
            poi_visits = self.POI_TIME_COUNTS[:, t]
            poi_cbg_visits = sparse.diags(poi_visits) @ self.POI_CBG_PROPORTIONS

            # Do IPF if cbg_day_prop_out is provided
            if self.cbg_day_prop_out is not None:
                day = int(t / 24)
                cbg_prop_out = self.cbg_day_prop_out[:, day]

                # rows are POIs, columns are CBGs.
                target_row_sums = poi_visits * (self.POI_CBG_PROPORTIONS @ np.ones(self.POI_CBG_PROPORTIONS.shape[1])) # POI sums. This is the same as poi_visits * np.sum(self.POI_CBG_PROPORTIONS, axis = 1) but it's 5x faster for totally unknown reasons
                target_col_sums = cbg_prop_out * self.CBG_SIZES # CBG sums
                target_col_sums = target_col_sums * np.sum(target_row_sums) / np.sum(target_col_sums) # Renormalize to match POI sums

                assert len(target_row_sums.shape) == 1
                assert len(target_col_sums.shape) == 1
                assert self.POI_CBG_PROPORTIONS.shape[0] == len(target_row_sums)
                assert self.POI_CBG_PROPORTIONS.shape[1] == len(target_col_sums)

                # The matrix starts row normalized
                zero_col_idxs = (poi_visits @ self.POI_CBG_PROPORTIONS) == 0
                zero_row_idxs = (target_row_sums == 0)

                for i in range(self.ipf_num_iter):
                    # Normalize cols (CBGs)
                    col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))
                    col_sums[zero_col_idxs] = 1
                    c = target_col_sums / col_sums
                    poi_cbg_visits = poi_cbg_visits @ sparse.diags(c)

                    if (i == self.ipf_num_iter - 1) and (self.ipf_final_match == 'cbg'):
                        # End col normalized
                        break

                    # Normalize rows (POIs)
                    row_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=1)))
                    row_sums[zero_row_idxs] = 1
                    r = target_row_sums / row_sums
                    poi_cbg_visits = sparse.diags(r) @ poi_cbg_visits


                self.poi_cbg_visit_history.append(poi_cbg_visits)

        self.num_active_pois = np.sum(poi_visits > 0)

        col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))
        self.cbg_num_out = col_sums

        # S x M = (M) * ((M x N) @ (S x N).T ).T
        poi_infection_rates = self.POI_FACTORS * (poi_cbg_visits @ cbg_densities.T).T
        self.num_poi_infection_rates_clipped = np.sum(poi_infection_rates > 1)
        if self.clip_poisson_approximation:
            poi_infection_rates = np.clip(poi_infection_rates, None, 1.0)

        # S x N = (S x N) * ((S x M) @ (M x N))
        cbg_mean_new_cases_from_poi = sus_frac * (poi_infection_rates @ poi_cbg_visits)

        if len(self.groups_to_track_num_cases_per_poi) > 0:
            self.cbg_num_cases_per_poi = []  # S x M x N
            for s in range(self.num_seeds):
                seed_sus_frac = sus_frac[s]  # 1 x N
                seed_poi_cbg_sus = poi_cbg_visits.multiply(seed_sus_frac)  # (1 x N) * (M x N) -> row will broadcast
                seed_poi_infection_rates = poi_infection_rates[s]  # 1 x M
                # num people from cbg infected at this poi
                seed_poi_cbg_infected = seed_poi_cbg_sus.transpose().multiply(seed_poi_infection_rates).transpose()  # M x N
                self.cbg_num_cases_per_poi.append(seed_poi_cbg_infected)

        self.num_cbgs_active_at_pois = np.sum(cbg_mean_new_cases_from_poi > 0)

        if self.stochastic:
            if self.debug:
                print(f'using poisson approx: expected new cases = {np.sum(cbg_mean_new_cases)}')
            num_cases_from_poi = np.random.poisson(cbg_mean_new_cases_from_poi)
            self.num_cbgs_with_clipped_poi_cases = np.sum(num_cases_from_poi > num_sus)
            # self.cbg_new_cases_from_poi = num_cases_from_poi
            self.cbg_new_cases_from_poi = np.clip(num_cases_from_poi, None, num_sus)

            num_sus_remaining = num_sus - self.cbg_new_cases_from_poi
            # num_sus_remaining = np.clip(num_sus - self.cbg_new_cases_from_poi, 0, None)

            self.cbg_new_cases_from_base = np.random.binomial(
                num_sus_remaining.astype(int),
                cbg_base_infection_rates)

            self.cbg_new_cases = self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base

            # keep track of whether clipping is doing scary stuff.
            self.clipping_monitor['num_base_infection_rates_clipped'].append(self.num_base_infection_rates_clipped)
            self.clipping_monitor['num_active_pois'].append(self.num_active_pois)
            self.clipping_monitor['num_poi_infection_rates_clipped'].append(self.num_poi_infection_rates_clipped)
            self.clipping_monitor['num_cbgs_active_at_pois'].append(self.num_cbgs_active_at_pois)
            self.clipping_monitor['num_cbgs_with_clipped_poi_cases'].append(self.num_cbgs_with_clipped_poi_cases)

        else:
            self.cbg_new_cases_from_poi = np.clip(cbg_mean_new_cases_from_poi, None, num_sus)
            num_sus_remaining = num_sus - self.cbg_new_cases_from_poi
            self.cbg_new_cases_from_base = num_sus_remaining * cbg_base_infection_rates
            self.cbg_new_cases = self.self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base
            self.num_cbgs_with_clipped_poi_cases = 0

        assert (self.cbg_new_cases <= num_sus).all()

    def get_new_infectious(self):
        # individuals leave L at a rate inversely proportional
        # to the latency period
        if self.stochastic:
            new_infectious = np.random.binomial(self.cbg_latent.astype(int), 1 / self.LATENCY_PERIOD)
            return new_infectious
        return self.cbg_latent * (1 / self.LATENCY_PERIOD)

    def get_new_removed(self):
        # individuals leave I at a rate inversely proportional
        # to the infectious period
        if self.stochastic:
            new_removed = np.random.binomial(self.cbg_infected.astype(int), 1 / self.INFECTIOUS_PERIOD)
            return new_removed
        return self.cbg_infected * (1 / self.INFECTIOUS_PERIOD)

    def format_floats(self, arr):
        '''
        Helper function that returns an array of floats with each float
        rounded to its nearest integer. This is useful when reporting the
        CBG disease states, so that print statements do not get too long.
        '''
        return [int(round(x)) for x in arr]

    def save(self, file):
        self.POI_CBG_PROPORTIONS = None
        self.cbg_day_prop_out = None
        self.POI_TIME_COUNTS = None
        self.get_new_cases = None
        self.POI_CBG_VISITS_LIST = None
        self.POI_AREAS = None
        self.DWELL_TIME_CORRECTION_FACTORS = None
        self.POI_FACTORS = None
        self.POI_SUBCATEGORY_TYPES = None
        pickle.dump(self, file, protocol=4) # https://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving
        file.close()
