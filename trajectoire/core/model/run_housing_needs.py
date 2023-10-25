# -*- coding: utf-8 -*-

# ========================================
# External imports
# ========================================

import pyam
import pandas as pd

from importlib import resources

# ========================================
# External CSTB imports
# ========================================


# ========================================
# Internal imports
# ========================================

from trajectoire.core.model.housing_needs import (
    calculate_housing_needs,
    calculate_construction_needs_from_population_growth,
    run_construction_pyam,
    calculate_combined_ghge_new_construction,
    calculate_cumulated_operational_ghge_new_construction,
)
from trajectoire.core.model.plotting_functions import (
    plot_scenarios_construction,
    plot_cumulative_ghge_area
)
from trajectoire import results
from trajectoire.config import data

# ========================================
# Constants
# ========================================

DATA_DIRECTORY_PATH = resources.files(data)
SCENARIOS_DIRECTORY_PATH = DATA_DIRECTORY_PATH / "scenarios"
CALIBRATION_DIRECTORY_PATH = DATA_DIRECTORY_PATH / "calibration"
RESULT_DIRECTORY_PATH = resources.files(results)
RESULT_HOUSING_DIRECTORY_PATH = RESULT_DIRECTORY_PATH / "results_housing_needs"

RESULT_HOUSING_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

# ========================================
# Variables
# ========================================


# ========================================
# Classes
# ========================================


# ========================================
# Functions
# ========================================


# ========================================
# Scripts
# ========================================

# === LOAD SCENARIOS ===
population = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "new_construction" / "population_scenarios.csv", sep=';')
FApC = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "new_construction" / "m2_per_capita_scenarios.csv", sep=';')
dem = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "demolition" / "demolition_scenarios_bdnb.csv", sep=';')
new = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "new_construction" / "new_surface_scenarios.csv", sep=';')
pct_ind_col_surface = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "new_construction" / "pct_ind_col_new_scenarios.csv", sep=';')
embodied_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "ghge" / "embodied_ghge_new_construction_scenarios.csv", sep=';')
operational_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH  / "ghge" / "operational_ghge_new_construction_scenarios.csv", sep=';')

## Fonctions de base de pyam
### List of available scenarios
print('population scenario: ', population.scenario)
print('m2/pers: ', FApC.scenario)
print('demolition: ', dem.scenario)
print('surface new construction: ', new.scenario)
print('%MI/LC en nouvelle surface: ', pct_ind_col_surface.scenario)
print('embodied ghge: ', embodied_ghge.scenario)
print('operational ghge: ', operational_ghge.scenario)


#### If we want to have a list of data or timeseries
population.data
population.timeseries


# CALCULATE HOUSING NEEDS AND CONSTRUCTION NEEDS FROM GROWTH
combined_housing_needs_iamdf = calculate_housing_needs(population, FApC,
                            population_scenarios=population.scenario, FApC_scenarios=FApC.scenario,
                            FApC_variable='shab/pers',
                            unit_name='m2')

combined_construction_needs_from_population_growth_iamdf = calculate_construction_needs_from_population_growth(combined_housing_needs_iamdf)
combined_construction_needs_from_population_growth_iamdf.to_csv(RESULT_HOUSING_DIRECTORY_PATH / 'construction_needs_from_demand_growth_iamdf.csv', sep=';')

# CALCULATE CONSTRUCTION SCENARIOS IN SURFACE AND DWELLING
## BAU
df_bau_construction = run_construction_pyam(combined_construction_needs_from_population_growth_iamdf,
                                         dem_scenario='HDS',
                                         construction_needs_scenario='Scenario central_BAU',
                                         pct_ind_col_surface_scenario='BAU',
                                         new_scenario='BAU',
                                         created_scenario_construction_name='BAU_construction')

df_bau_construction.timeseries().reset_index().to_csv(RESULT_HOUSING_DIRECTORY_PATH / "iamdf_bau_construction.csv", sep=';', index=False)

## S1
# df_s1_construction = run_construction_pyam(combined_construction_needs_from_population_growth_iamdf,
#                                         dem_scenario='MDS',
#                                         construction_needs_scenario='Population basse_Negawatt',
#                                         pct_ind_col_surface_scenario='BAU',
#                                         new_scenario='S1',
#                                         created_scenario_construction_name='S1_construction')

## S2
df_s2_construction = run_construction_pyam(combined_construction_needs_from_population_growth_iamdf,
                                        dem_scenario='MDS',
                                        construction_needs_scenario='Scenario central_Negawatt',
                                        pct_ind_col_surface_scenario='S2',
                                        new_scenario='S2',
                                        created_scenario_construction_name='S2_construction')

## Combine all in a single Iamdf
combined_construction_df = pd.concat([df_bau_construction.data, df_s2_construction.data], ignore_index=True)
combined_construction_iamdf = pyam.IamDataFrame(combined_construction_df)
combined_construction_iamdf.to_csv(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_iamdf.csv", sep=';')

fig_construction_res_dwelling = plot_scenarios_construction(combined_construction_iamdf, variable='Residential_dwelling', unit='Number of dwelling', title='Residential dwelling in BAU and S2', ylabel='Number of dwellings')
fig_construction_col_dwelling = plot_scenarios_construction(combined_construction_iamdf, variable='Collective_dwelling', unit='Number of dwelling', title='Collective dwelling in BAU and S2', ylabel='Number of dwellings')
fig_construction_ind_dwelling = plot_scenarios_construction(combined_construction_iamdf, variable='Individual_dwelling', unit='Number of dwelling', title='Individual dwelling in BAU and S2', ylabel='Number of dwellings')

fig_construction_res_surface = plot_scenarios_construction(combined_construction_iamdf, variable='Residential_dwelling', unit='Surface in m2', title='Residential surface in BAU and S2', ylabel='m2')
fig_construction_col_surface = plot_scenarios_construction(combined_construction_iamdf, variable='Collective_dwelling', unit='Surface in m2', title='Collective surface in BAU and S2', ylabel='m2')
fig_construction_ind_surface = plot_scenarios_construction(combined_construction_iamdf, variable='Individual_dwelling', unit='Surface in m2', title='Individual surface in BAU and S2', ylabel='m2')

# CALCULATE EMBODIED GHGE
combined_construction_embodied_iamdf = calculate_combined_ghge_new_construction(combined_construction_iamdf, embodied_ghge)
combined_construction_embodied_iamdf.to_csv(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_embodied_iamdf.csv", sep=';')

fig_construction_embodied_res = plot_scenarios_construction(combined_construction_embodied_iamdf, variable='Residential_dwelling', unit='MtCO2eq', title='New residential embodied GHGE', figsize=(10,8), ylabel='MtCO2eq')
fig_construction_embodied_col = plot_scenarios_construction(combined_construction_embodied_iamdf, variable='Collective_dwelling', unit='MtCO2eq', title='New collective embodied GHGE', figsize=(10,8), ylabel='MtCO2eq')
fig_construction_embodied_ind = plot_scenarios_construction(combined_construction_embodied_iamdf, variable='Individual_dwelling', unit='MtCO2eq', title='New individual embodied GHGE', figsize=(10,8), ylabel='MtCO2eq')

fig_construction_embodied_res_cum = plot_cumulative_ghge_area(combined_construction_embodied_iamdf, scenario_pattern='*', variable='Residential_dwelling', title='New residential cumulated embodied GHGE', figsize=(10,8))
fig_construction_embodied_col_cum = plot_cumulative_ghge_area(combined_construction_embodied_iamdf, scenario_pattern='*', variable='Collective_dwelling', title='New collective cumulated embodied GHGE', figsize=(10,8))
fig_construction_embodied_ind_cum = plot_cumulative_ghge_area(combined_construction_embodied_iamdf, scenario_pattern='*', variable='Individual_dwelling', title='New individual cumulated embodied GHGE', figsize=(10,8))


# CALCULATE OPERATIONAL GHGE
## Yearly values
combined_construction_operational_yearly_iamdf = calculate_combined_ghge_new_construction(combined_construction_iamdf, operational_ghge)
combined_construction_operational_yearly_iamdf.to_csv(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_operational_yearly_iamdf.csv", sep=';')


## Cumulated values
combined_construction_operational_cumulated_iamdf = calculate_cumulated_operational_ghge_new_construction(combined_construction_iamdf, operational_ghge)
combined_construction_operational_cumulated_iamdf.to_csv(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_operational_cumulated_iamdf.csv", sep=';')

fig_construction_operational_cumulated_res = plot_scenarios_construction(combined_construction_operational_cumulated_iamdf, variable='Residential_dwelling', unit='MtCO2eq', title='Residential dwelling cumulated operational GHGE', figsize=(10,8))
fig_construction_operational_cumulated_col = plot_scenarios_construction(combined_construction_operational_cumulated_iamdf, variable='Collective_dwelling', unit='MtCO2eq', title='Collective dwelling cumulated operational GHGE', figsize=(10,8))
fig_construction_operational_cumulated_ind = plot_scenarios_construction(combined_construction_operational_cumulated_iamdf, variable='Individual_dwelling', unit='MtCO2eq', title='Individual dwelling cumulated operational GHGE', figsize=(10,8))

