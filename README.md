# covid-mobility

Code to generate results in "[Mobility network models of COVID-19 explain inequities and inform reopening](https://www.medrxiv.org/content/10.1101/2020.06.15.20131979v1)" (2020) by Serina Y Chang, Emma Pierson, Pang Wei Koh, Jaline Gerardin, Beth Redbird, David Grusky, and Jure Leskovec. 

## Regenerating results

1. **Setting up virtualenv**. Our code is run in a conda environment, with all analysis performed on a Linux Ubuntu system. You can set up this environment by running `conda env create --prefix YOUR_PATH_HERE --file safegraph_env_v3.yml`. Once you have set up the environment, activate it prior to running any code by running `source YOUR_PATH_HERE/bin/activate`. 

2. **Downloading datasets**.

    - SafeGraph data is freely available to researchers, non-profits, and governments through the [SafeGraph COVID-19 Data Consortium](https://www.safegraph.com/covid-19-data-consortium). As described in the Methods section, we use v1 of the Weekly Patterns Data from March 1 - May 2 2020; Monthly Patterns data from January 2019 - February 2020; and Social Distancing Metrics from March 1 2020 - May 2, 2020. 
    
    - We use case and death count data from *The New York Times*, available [here](https://github.com/nytimes/covid-19-data). While the *The New York Times* updates the data regularly, results in our paper are generated using case and death counts through May 9, 2020. 
    
    - Census data comes from the American Community Survey. Census block group shapefiles, with linked data from the 5-year 2013-2017 ACS, are available [here](https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip). (We note that, as described in the Methods, we use the 1-year 2018 estimates for the current population of each census block group.) The mapping from counties to MSAs is available [here](https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls). 
    
    - We use Google mobility data as a validation of SafeGraph data quality, available [here](https://google.com/covid19/mobility/). 

3. **Data processing**. We process SafeGraph Patterns data and combine it with Census data using `process_safegraph_data.ipynb`. This notebook takes a while to run, and the files are large; we suggest running it in a screen session or similar on a cluster using a command like `jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook process_safegraph_data.ipynb`. As a general note, our code is currently designed to run on our servers (eg, it makes reference to our specific filepaths). We wanted to provide a copy of the codebase as quickly as possible to reviewers, for maximum transparency, but will further clean up the code in coming weeks so that it is easier for others to use. 

4. **Running models.**
    - Models are run using `model_experiments.py`. The experimental pipeline has several steps which must be run in a particular order. Running all the models described in the paper is computationally expensive. Specifically, most experiments in the paper were performed using a server with 288 threads and 12 TB RAM; saving the models required several terabytes of disk space. We highlight steps that are particularly computationally expensive. 
    - **a. Generate the hourly visit matrices by running IPFP**. Run `python model_experiments.py run_many_models_in_parallel just_save_ipf_output`. This will start one job for each MSA which generates the hourly visit matrices through the iterative proportional fitting procedure (IPFP). 
    - **b. Determine plausible ranges for model parameters over which to conduct grid search.**. Run `python model_experiments.py run_many_models_in_parallel calibrate_r0`. This will start several hundred jobs.
    - **c. Conduct grid search to find models which best fit case counts.**. Run `python model_experiments.py run_many_models_in_parallel normal_grid_search`. This is a computationally expensive step which will fit thousands of models; even starting all the models may take several hours. 
    - The remaining experiments rely on having grid search completed, since they use the best-fit model parameters. However, once grid search is performed, they can be run in any order. Be sure to change the variable `min_timestring_to_load_best_fit_models_from_grid_search` so that it is equivalent to the timestring for the first grid search experiment. All experiments can be run using the same call signature as above: `python model_experiments.py run_many_models_in_parallel EXPERIMENT_NAME`. The specific experiments are: 
        - `test_interventions`: This tests the effects of reopening each POI subcategory. This is computationally expensive because it runs one model for each category, MSA, and best-fit model parameter setting; in total, this is several thousand models. 
        - `test_retrospective_counterfactuals`: This simulates the impacts of various counterfactuals of past mobility reduction on infection outcomes. This is moderately expensive computationally (several hundred jobs), because it runs one model for each counterfactual setting, MSA, and best-fit model parameter setting.
        - `test_max_capacity_clipping`: This tests the effects of partial reopening by ''clipping'' each POI's visits to a fraction of its maximum capacity (or occupancy).  This will start around 1000 jobs, running one model for each level of clipping, MSA, and best-fit model parameter setting.
        - `test_uniform_proportion_of_full_reopening`: This tests the effects of partial reopening by uniformly reducing visits to each POI from their activity levels in early March. This will also start around 1000 jobs, running one model for each level of reopening, MSA, and best-fit model parameter setting.
        - `rerun_best_models_and_save_cases_per_poi`: This reruns the best-fit models for each MSA and saves the expected number of infectons that occurred at each POI on each day. We do not save infections per POI by default, because this takes up too much space and slows down the simulation process. This is the least computationally expensive of the experiments, just running each best-fit model parameter setting once.

5. **Analyzing models and generating results for paper**. Once models have been run, figures and results in the paper can be reproduced by running `make_figures.ipynb` and `supplementary_analyses.ipynb`. See below for details.

## Files

**covid_constants_and_util.py**: Constants and general utility methods. 

**disease_model.py**: Implements the disease model on the mobility network. 

**helper_methods_for_aggregate_data_analysis.py**: Various helper methods used in data processing and throughout the analysis. 

**make_figures.ipynb**: Once the models have been run, reproduces the main figures (Figures 1-3) and all of the Extended Data and SI figures and tables that are directly related to the main figures (e.g., results for all metro areas, in the case that the main figure only highlights one metro area).

**make_network_map.ipynb**: Constructs the POI-CBG spatial maps in Figure 1a.

**model_experiments.py**: Runs models for the experiments described in the paper. 

**process_safegraph_data.ipynb**: Processes the raw SafeGraph data. 

**safegraph_env_v3.yml**: Used to set up the conda environment. 

**supplementary_analyses.ipynb**: Once the models have been run, reproduces the remaining Extended Data and SI figures and tables, including sensitivity analyses and checks for parameter identifiability.

**test_google_correlation.ipynb**: Tests the correlation between Google and SafeGraph mobility data.
