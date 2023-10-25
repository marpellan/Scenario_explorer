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
    calculate_combined_ghge_new_construction,
)

from trajectoire.core.model.plotting_functions import (
    plot_reno_surface_scenarios,
    plot_reno_embodied_ghge_scenarios,
    plot_stock_surface_scenarios,
    plot_stock_operational_ghge_scenarios,
    plot_cumulative_ghge_area,
    plot_cumulative_ghge_lines,
    plot_operational_stock_dpe,
    plot_cumulative_operational_stock_dpe,
    plot_embodied_all_stock,
    plot_cumulative_all_stock_embodied,
    plot_operational_all_stock_dpe,
    plot_cumulative_operational_all_stock_dpe,
    wlc_graph,
    wlc_cum_graph,
    wlc_graph_sensibility,
    wlc_graph_sensibility_cum
)

from trajectoire.core.model.dynamic_epc_matrix import (
    run_dynamic_simulation,
    calculate_jumps_epc_label,
    create_iamdf_reno,
    create_iamdf_stock,
    add_residential_dwelling_reno,
    add_residential_dwelling_stock,
    group_saut_dpe_reno,
    group_dpe_stock,
    calculate_combined_embodied_reno_iamdf,
    calculate_combined_operational_stock_iamdf,
    embodied_all_stock,
    operational_all_stock,
    embodied_all_stock_combined,
    operational_all_stock_combined,
    drop_inconsistent_operational_scenarios,
    drop_inconsistent_embodied_scenarios
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
RESULT_EPC_MATRIX_DIRECTORY_PATH = RESULT_DIRECTORY_PATH / "results_epc_matrix"
RESULT_HOUSING_DIRECTORY_PATH = RESULT_DIRECTORY_PATH / "results_housing_needs"


RESULT_EPC_MATRIX_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

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

# === LOAD STOCK DATA ===
#df_stock_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH + / "dwelling_stock_per_epc" / "dwelling_stock_before_renovation.csv", sep=';', index_col=[0,1])
df_stock_col_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "dwelling_stock_before_renovation_col.csv", sep=';', index_col=[0,1])
df_stock_ind_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "dwelling_stock_before_renovation_ind.csv", sep=';', index_col=[0,1])

# === LOAD SCENARIOS ===
### Renovation and demolition pace in pyam format
renovation = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "renovation" / "renovation_scenarios_bdnb.csv", sep=';')
demolition = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "demolition" / "demolition_scenarios_bdnb.csv", sep=';')

### Operational and embodied GHGE ratios in kgCO2eq/m2
operational_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "operational_ghge_renovation_scenarios.csv", sep=';')
embodied_ghge_reno = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_renovation_scenarios.csv", sep=';')
embodied_ghge_dem = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_demolition_scenarios.csv", sep=';')

### List of available scenarios
print('renovation scenario: ', renovation.scenario)
print('demolition scenario: ', demolition.scenario)
print('operational GHGE scenario: ', operational_ghge.scenario)
print('embodied GHGE scenario for renovation: ', embodied_ghge_reno.scenario)
print('embodied GHGE scenario for demolition: ', embodied_ghge_dem.scenario)

### Pivot table from the High Renovation Scenarios
pt_col_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_hrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_hrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the Medium Renovation Scenarios
pt_col_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_mrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_mrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the High Demolition Scenarios
pt_col_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the Medium Demolition Scenarios
pt_col_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)


# === RUN SIMULATION YEAR BY YEAR AND UPDATE THE RELEVANT PARTS ===
## Collective dwelling
### HRS and MRS with HDS
updated_df_stock_hrs_rising_hds_col, changes_reno_lists_hrs_rising_col, changes_dem_lists_hds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='HRS_rising', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_hrs_normal_hds_col, changes_reno_lists_hrs_normal_col, changes_dem_lists_hds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='HRS_normal', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_mrs_rising_hds_col, changes_reno_lists_mrs_rising_col, changes_dem_lists_hds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_rising', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_mrs_normal_hds_col, changes_reno_lists_mrs_normal_col, changes_dem_lists_hds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_normal', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

### HRS and MRS with MDS
updated_df_stock_hrs_rising_mds_col, changes_reno_lists_hrs_rising_col, changes_dem_lists_mds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='HRS_rising', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_hrs_normal_mds_col, changes_reno_lists_hrs_normal_col, changes_dem_lists_mds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='HRS_normal', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_mrs_rising_mds_col, changes_reno_lists_mrs_rising_col, changes_dem_lists_mds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_rising', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_mrs_normal_mds_col, changes_reno_lists_mrs_normal_col, changes_dem_lists_mds_col = run_dynamic_simulation(
    pt_col_reno_hrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_normal', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

## Individual dwelling
### HRS and MRS with HDS
updated_df_stock_hrs_rising_hds_ind, changes_reno_lists_hrs_rising_ind, changes_dem_lists_hds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='HRS_rising', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_hrs_normal_hds_ind, changes_reno_lists_hrs_normal_ind, changes_dem_lists_hds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='HRS_normal', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_mrs_rising_hds_ind, changes_reno_lists_mrs_rising_ind, changes_dem_lists_hds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_rising', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_mrs_normal_hds_ind, changes_reno_lists_mrs_normal_ind, changes_dem_lists_hds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_normal', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

### HRS and MRS with MDS
updated_df_stock_hrs_rising_mds_ind, changes_reno_lists_hrs_rising_ind, changes_dem_lists_mds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='HRS_rising', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_hrs_normal_mds_ind, changes_reno_lists_hrs_normal_ind, changes_dem_lists_mds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='HRS_normal', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_mrs_rising_mds_ind, changes_reno_lists_mrs_rising_ind, changes_dem_lists_mds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_rising', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_mrs_normal_mds_ind, changes_reno_lists_mrs_normal_ind, changes_dem_lists_mds_ind = run_dynamic_simulation(
    pt_ind_reno_hrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_normal', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)


# === GROUP RENOVATION BY JUMPS ===
## Collective dwelling
df_jumps_reno_hrs_rising_col = calculate_jumps_epc_label(changes_reno_lists_hrs_rising_col)
df_jumps_reno_hrs_normal_col = calculate_jumps_epc_label(changes_reno_lists_hrs_normal_col)
df_jumps_reno_mrs_rising_col = calculate_jumps_epc_label(changes_reno_lists_mrs_rising_col)
df_jumps_reno_mrs_normal_col = calculate_jumps_epc_label(changes_reno_lists_mrs_normal_col)

iamdf_reno_hrs_rising_col = create_iamdf_reno(df_jumps_reno_hrs_rising_col,
                                   model_name='CSTB',
                                   scenario_name='HRS_rising',
                                   variable_name='Collective_dwelling')

iamdf_reno_hrs_normal_col = create_iamdf_reno(df_jumps_reno_hrs_normal_col,
                                   model_name='CSTB',
                                   scenario_name='HRS_normal',
                                   variable_name='Collective_dwelling')

iamdf_reno_mrs_rising_col = create_iamdf_reno(df_jumps_reno_mrs_rising_col,
                                   model_name='CSTB',
                                   scenario_name='MRS_rising',
                                   variable_name='Collective_dwelling')

iamdf_reno_mrs_normal_col = create_iamdf_reno(df_jumps_reno_mrs_normal_col,
                                   model_name='CSTB',
                                   scenario_name='MRS_normal',
                                   variable_name='Collective_dwelling')

combined_dfs_reno_col = pd.concat([iamdf_reno_hrs_rising_col.data, iamdf_reno_hrs_normal_col.data, iamdf_reno_mrs_rising_col.data, iamdf_reno_mrs_normal_col.data], ignore_index=True)
combined_iamdfs_reno_col = pyam.IamDataFrame(combined_dfs_reno_col)


## Individual dwelling
df_jumps_reno_hrs_rising_ind = calculate_jumps_epc_label(changes_reno_lists_hrs_rising_ind)
df_jumps_reno_hrs_normal_ind = calculate_jumps_epc_label(changes_reno_lists_hrs_normal_ind)
df_jumps_reno_mrs_rising_ind = calculate_jumps_epc_label(changes_reno_lists_mrs_rising_ind)
df_jumps_reno_mrs_normal_ind = calculate_jumps_epc_label(changes_reno_lists_mrs_normal_ind)

iamdf_reno_hrs_rising_ind = create_iamdf_reno(df_jumps_reno_hrs_rising_ind,
                                   model_name='CSTB',
                                   scenario_name='HRS_rising',
                                   variable_name='Individual_dwelling')

iamdf_reno_hrs_normal_ind = create_iamdf_reno(df_jumps_reno_hrs_normal_ind,
                                   model_name='CSTB',
                                   scenario_name='HRS_normal',
                                   variable_name='Individual_dwelling')

iamdf_reno_mrs_rising_ind = create_iamdf_reno(df_jumps_reno_mrs_rising_ind,
                                   model_name='CSTB',
                                   scenario_name='MRS_rising',
                                   variable_name='Individual_dwelling')

iamdf_reno_mrs_normal_ind = create_iamdf_reno(df_jumps_reno_mrs_normal_ind,
                                   model_name='CSTB',
                                   scenario_name='MRS_normal',
                                   variable_name='Individual_dwelling')

combined_dfs_reno_ind = pd.concat([iamdf_reno_hrs_rising_ind.data, iamdf_reno_hrs_normal_ind.data, iamdf_reno_mrs_rising_ind.data, iamdf_reno_mrs_normal_ind.data], ignore_index=True)
combined_iamdfs_reno_ind = pyam.IamDataFrame(combined_dfs_reno_ind)


### WE COMBINE COL AND IND IAMDF_RENO AND ADD RESIDENTIAL DWELLING AS THE SUM OF COL AND IND
combined_dfs_reno = pd.concat([combined_iamdfs_reno_col.data, combined_iamdfs_reno_ind.data], ignore_index=True)
combined_iamdfs_reno = pyam.IamDataFrame(combined_dfs_reno)
combined_iamdfs_reno_res = add_residential_dwelling_reno(combined_iamdfs_reno)

### We add the cumulated renovation, no matter the number of jumps
combined_iamdfs_reno_res = group_saut_dpe_reno(combined_iamdfs_reno)
combined_iamdfs_reno_res.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_reno_surface_iamdf.csv", sep=';')

# === CALCULATE EMBODIED GHGE OF RENOVATIONS
combined_reno_embodied_iamdf = calculate_combined_embodied_reno_iamdf(combined_iamdfs_reno, embodied_ghge_reno)
combined_reno_embodied_iamdf = group_saut_dpe_reno(combined_reno_embodied_iamdf)
combined_reno_embodied_iamdf.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_reno_embodied_iamdf.csv", sep=';')

### FIG TO REPRESENT GHGE TRAJECTORIES
fig_reno_embodied_hrs = plot_reno_embodied_ghge_scenarios(combined_reno_embodied_iamdf, scenario='HRS*', variable='Residential_dwelling', saut_classe_dpe='sum', title='High Renovation Scenarios embodied GHGE', figsize=(10,8))
fig_reno_embodied_mrs = plot_reno_embodied_ghge_scenarios(combined_reno_embodied_iamdf, scenario='MRS*', variable='Residential_dwelling', saut_classe_dpe='sum', title='Middle Renovation Scenarios embodied GHGE', figsize=(10,8))

### FIG TO REPRESENT CUMULATIVE GHGE
fig_reno_embodied_hrs_cumulated_area = plot_cumulative_ghge_area(combined_reno_embodied_iamdf, scenario_pattern='HRS*', variable='Residential_dwelling', title='High Renovation Scenarios cumulated embodied GHGE', figsize=(10,8))
fig_reno_embodied_mrs_cumulated_area = plot_cumulative_ghge_area(combined_reno_embodied_iamdf, scenario_pattern='MRS*', variable='Residential_dwelling', title='Middle Renovation Scenarios cumulated embodied GHGE', figsize=(10,8))

#fig_reno_embodied_hrs_cumulated_lines = plot_cumulative_ghge_lines(combined_reno_embodied_iamdf, scenario_pattern='HRS*', variable='Residential_dwelling', title='High Renovation Scenarios cumulated embodied GHGE', figsize=(10,8))
#fig_reno_embodied_mrs_cumulated_lines = plot_cumulative_ghge_lines(combined_reno_embodied_iamdf, scenario_pattern='MRS*', variable='Residential_dwelling', title='Middle Renovation Scenarios cumulated embodied GHGE', figsize=(10,8))

# == CREATE IAMDF STOCK
## Collective dwelling
### HRS/MRS with HDS
iamdf_stock_hrs_rising_hds_col = create_iamdf_stock(
                        updated_df_stock_hrs_rising_hds_col,
                         scenario_name='HRS_rising_HDS',
                         variable_name='Collective_dwelling')

iamdf_stock_hrs_normal_hds_col = create_iamdf_stock(
                        updated_df_stock_hrs_normal_hds_col,
                        scenario_name='HRS_normal_HDS',
                        variable_name='Collective_dwelling')

iamdf_stock_mrs_rising_hds_col = create_iamdf_stock(
                        updated_df_stock_mrs_rising_hds_col,
                        scenario_name='MRS_rising_HDS',
                        variable_name='Collective_dwelling')

iamdf_stock_mrs_normal_hds_col = create_iamdf_stock(
                        updated_df_stock_mrs_normal_hds_col,
                        scenario_name='MRS_normal_HDS',
                        variable_name='Collective_dwelling')

### HRS/MRS with MDS
iamdf_stock_hrs_rising_mds_col = create_iamdf_stock(
                        updated_df_stock_hrs_rising_mds_col,
                         scenario_name='HRS_rising_MDS',
                         variable_name='Collective_dwelling')

iamdf_stock_hrs_normal_mds_col = create_iamdf_stock(
                        updated_df_stock_hrs_normal_mds_col,
                        scenario_name='HRS_normal_MDS',
                        variable_name='Collective_dwelling')

iamdf_stock_mrs_rising_mds_col = create_iamdf_stock(
                        updated_df_stock_mrs_rising_mds_col,
                        scenario_name='MRS_rising_MDS',
                        variable_name='Collective_dwelling')

iamdf_stock_mrs_normal_mds_col = create_iamdf_stock(
                        updated_df_stock_mrs_normal_mds_col,
                        scenario_name='MRS_normal_MDS',
                        variable_name='Collective_dwelling')

## Individual dwelling
### HRS/MRS with HDS
iamdf_stock_hrs_rising_hds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_rising_hds_ind,
                         scenario_name='HRS_rising_HDS',
                         variable_name='Individual_dwelling')

iamdf_stock_hrs_normal_hds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_normal_hds_ind,
                        scenario_name='HRS_normal_HDS',
                        variable_name='Individual_dwelling')

iamdf_stock_mrs_rising_hds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_rising_hds_ind,
                        scenario_name='MRS_rising_HDS',
                        variable_name='Individual_dwelling')

iamdf_stock_mrs_normal_hds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_normal_hds_ind,
                        scenario_name='MRS_normal_HDS',
                        variable_name='Individual_dwelling')

### HRS/MRS with MDS
iamdf_stock_hrs_rising_mds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_rising_mds_ind,
                         scenario_name='HRS_rising_MDS',
                         variable_name='Individual_dwelling')

iamdf_stock_hrs_normal_mds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_normal_mds_ind,
                        scenario_name='HRS_normal_MDS',
                        variable_name='Individual_dwelling')

iamdf_stock_mrs_rising_mds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_rising_mds_ind,
                        scenario_name='MRS_rising_MDS',
                        variable_name='Individual_dwelling')

iamdf_stock_mrs_normal_mds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_normal_mds_ind,
                        scenario_name='MRS_normal_MDS',
                        variable_name='Individual_dwelling')


combined_iamdfs_stock = pd.concat([iamdf_stock_hrs_rising_hds_col.data,
                               iamdf_stock_hrs_normal_hds_col.data,
                               iamdf_stock_mrs_rising_hds_col.data,
                               iamdf_stock_mrs_normal_hds_col.data,
                               iamdf_stock_hrs_rising_mds_col.data,
                               iamdf_stock_hrs_normal_mds_col.data,
                               iamdf_stock_mrs_rising_mds_col.data,
                               iamdf_stock_mrs_normal_mds_col.data,
                               iamdf_stock_hrs_rising_hds_ind.data,
                               iamdf_stock_hrs_normal_hds_ind.data,
                               iamdf_stock_mrs_rising_hds_ind.data,
                               iamdf_stock_mrs_normal_hds_ind.data,
                               iamdf_stock_hrs_rising_mds_ind.data,
                               iamdf_stock_hrs_normal_mds_ind.data,
                               iamdf_stock_mrs_rising_mds_ind.data,
                               iamdf_stock_mrs_normal_mds_ind.data], ignore_index=True)

combined_iamdfs_stock = pyam.IamDataFrame(combined_iamdfs_stock)

### We add the number of residential dwelling
combined_iamdfs_stock_res = add_residential_dwelling_stock(combined_iamdfs_stock)
combined_iamdfs_stock_res.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_stock_surface_iamdf.csv", sep=';')


# == CALCULATE OPERATIONAL GHGE OF THE STOCK AFTER RENOVATION AND DEMOLITION ==
combined_iamdfs_stock_operational = calculate_combined_operational_stock_iamdf(combined_iamdfs_stock, operational_ghge)

### We add the cumulated, no matter the DPE
combined_iamdfs_stock_operational = group_dpe_stock(combined_iamdfs_stock_operational)
combined_iamdfs_stock_operational.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_stock_operational_iamdf.csv", sep=';')

### FIGS REMAINING STOCK OPERATIONAL GHGE FOR MULTIPLE SCENARIOS, GOOD FOR SENSIBILITY ANALYSIS
fig_stock_operational_hrs = plot_stock_operational_ghge_scenarios(combined_iamdfs_stock_operational, scenario='HRS*', variable='Residential_dwelling', dpe='sum', title='High Renovation Scenarios Operational GHGE', figsize=(10,8))
fig_stock_operational_mrs = plot_stock_operational_ghge_scenarios(combined_iamdfs_stock_operational, scenario='MRS*', variable='Residential_dwelling', dpe='sum', title='Middle Renovation Scenarios Operational GHGE', figsize=(10,8))

### FIGS REMAINING STOCK OPERATIONAL GHGE DPE FOR A PARTICULAR SCENARIO, BETTER LOOKING
#fig_stock_operational_dpe = plot_operational_stock_dpe(combined_iamdfs_stock_operational, scenario_pattern='HRS_rising_HDS_Quarter', variable='Residential_dwelling', title='Operational GHGE', figsize=(12, 8))
#fig_stock_operational_dpe_cumulated = plot_cumulative_operational_stock_dpe(combined_iamdfs_stock_operational, scenario_pattern='HRS_rising_HDS_Quarter', variable='Residential_dwelling', title='Cumulative operational GHGE', figsize=(12, 8))

# == CALCULATE EMBODIED GHGE OF DEMOLITION
combined_demolition_embodied_iamdf = calculate_combined_ghge_new_construction(demolition, embodied_ghge_dem)
combined_demolition_embodied_iamdf.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_demolition_embodied_iamdf.csv", sep=';')


# == COMBINE EMBODIED GHGE OF ALL THE STOCK, E.G. NEW CONSTRUCTION, RENOVATION & DEMOLITION ==
combined_construction_embodied_iamdf = pyam.IamDataFrame((RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_embodied_iamdf.csv"), sep=';')

embodied_bau = embodied_all_stock(
                            combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf,
                            created_scenario_name='BAU',
                            construction_scenario='BAU_construction_BAU_RE2020_upfront',
                            renovation_scenario='MRS_rising_BAU_RE2020_WLC',
                            demolition_scenario='HDS_Constant_WLC',
                            construction_variable='Residential_dwelling',
                            renovation_variable='Residential_dwelling',
                            demolition_variable='Residential_dwelling')

embodied_optimist = embodied_all_stock(
                            combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf,
                            created_scenario_name='Sufficiency',
                            construction_scenario='S2_construction_Optimist_RE2020_upfront',
                            renovation_scenario='HRS_rising_Optimist_RE2020_WLC',
                            demolition_scenario='MDS_Constant_WLC',
                            construction_variable='Residential_dwelling',
                            renovation_variable='Residential_dwelling',
                            demolition_variable='Residential_dwelling')

embodied_bau.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "embodied_bau.csv", sep=';')
embodied_optimist.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "embodied_optimistic.csv", sep=';')

fig_embodied_bau = plot_embodied_all_stock(embodied_bau,
                      title="Embodied GHGE - BAU")

fig_embodied_bau_cum = plot_cumulative_all_stock_embodied(embodied_bau,
                      title="Cumulated embodied GHGE - BAU")

fig_embodied_optimist = plot_embodied_all_stock(embodied_optimist,
                      title="Embodied GHGE - Sufficiency")

fig_embodied_optimist_cum = plot_cumulative_all_stock_embodied(embodied_optimist,
                      title="Cumulated embodied GHGE - Sufficiency")

## To compute all combination possible

combined_embodied_iamdf = embodied_all_stock_combined(combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf)
combined_embodied_iamdf = drop_inconsistent_embodied_scenarios(combined_embodied_iamdf)
combined_embodied_iamdf.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_embodied_all_stock_iamdf.csv", sep=';')

# == COMBINE OPERATIONAL GHGE OF ALL THE STOCK, E.G. NEW CONSTRUCTION, RENOVATION & DEMOLITION ==
combined_construction_operational_iamdf = pyam.IamDataFrame((RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_operational_cumulated_iamdf.csv"), sep=';')

operational_bau = operational_all_stock(
                            combined_construction_operational_iamdf, combined_iamdfs_stock_operational,
                            construction_scenario='BAU_construction_Constant',
                            remaining_scenario='MRS_rising_HDS_Constant',
                            created_scenario_name='BAU')

operational_optimist = operational_all_stock(
                            combined_construction_operational_iamdf, combined_iamdfs_stock_operational,
                            construction_scenario='S2_construction_Quarter',
                            remaining_scenario='HRS_rising_MDS_Quarter',
                            created_scenario_name='Optimistic')

## To compute all combination possible

combined_operational_iamdf = operational_all_stock_combined(combined_construction_operational_iamdf, combined_iamdfs_stock_operational)
combined_operational_iamdf = drop_inconsistent_operational_scenarios(combined_operational_iamdf)
combined_operational_iamdf.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "combined_operational_all_stock_iamdf.csv", sep=';')

## WHY IS IT WORKING ONLY BY SAVING IT AND RECALLING IT ?!
operational_bau.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "operational_bau.csv", sep=';')
operational_optimist.to_csv(RESULT_EPC_MATRIX_DIRECTORY_PATH / "operational_optimistic.csv", sep=';')

operational_bau = pyam.IamDataFrame((RESULT_EPC_MATRIX_DIRECTORY_PATH / "operational_bau.csv"), sep=';')
operational_optimist = pyam.IamDataFrame((RESULT_EPC_MATRIX_DIRECTORY_PATH / "operational_optimistic.csv"), sep=';')

fig_operational_bau = plot_operational_all_stock_dpe(operational_bau,
                                                     scenario_pattern='BAU',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - BAU',
                                                     figsize=(10,8))

fig_operational_bau_cum = plot_cumulative_operational_all_stock_dpe(operational_bau,
                                                                scenario_pattern='BAU',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - BAU',
                                                                figsize=(10,8))

fig_operational_optimistic = plot_operational_all_stock_dpe(operational_optimist,
                                                     scenario_pattern='Optimistic',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - Sufficiency',
                                                     figsize=(10,8))

fig_operational_operational_cum = plot_cumulative_operational_all_stock_dpe(operational_optimist,
                                                                scenario_pattern='Optimistic',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - Sufficiency',
                                                                figsize=(10,8))


# == Whole Life Cycle GHGE ==
graph_2050 = wlc_graph(operational_bau, operational_optimist, embodied_bau, embodied_optimist, year=2050)
graph_cum = wlc_cum_graph(operational_bau, operational_optimist, embodied_bau, embodied_optimist)

# == Sensibility graphs ==


graph_sensibility_2030 = wlc_graph_sensibility(combined_operational_iamdf, combined_embodied_iamdf, year=2030)
graph_sensibility_2040 = wlc_graph_sensibility(combined_operational_iamdf, combined_embodied_iamdf, year=2040)
graph_sensibility_2050 = wlc_graph_sensibility(combined_operational_iamdf, combined_embodied_iamdf, year=2050)

graph_sensibility_cum = wlc_graph_sensibility_cum(combined_operational_iamdf, combined_embodied_iamdf)

