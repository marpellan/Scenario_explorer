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
    plot_operational_all_stock_nrj,
    plot_cumulative_operational_all_stock_dpe,
    plot_cumulative_operational_all_stock_nrj,
    wlc_graph_compare_scenarios,
    wlc_graph_sensibility,
    wlc_graph_sensibility_cum,
    plot_stock_nrj_twh,
    plot_range_operational_ghge,
    plot_range_embodied_ghge
)


from trajectoire.core.model.dynamic_stock import (
    update_pivot_table_reno,
    update_pivot_table_dem,
    update_dwelling_stock_nrj,
    run_dynamic_simulation_nrj,
    calculate_jumps_epc_label,
    create_iamdf_reno,
    add_residential_dwelling_reno,
    create_iamdf_stock,
    add_residential_dwelling_stock,
    group_saut_dpe_reno,

    calculate_combined_operational_stock_iamdf,
    calculate_combined_embodied_reno_iamdf,
    calculate_combined_ghge_new_construction,
    embodied_all_stock,
    embodied_all_stock_combined,
    drop_inconsistent_embodied_scenarios,
    operational_all_stock,
    operational_all_stock_combined,
    drop_inconsistent_operational_scenarios,
    drop_inconsistent_nrj_scenarios
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
RESULT_NEW_NRJ_PATH = RESULT_DIRECTORY_PATH / "results_new_nrj"
RESULT_NEW_GES_PATH = RESULT_DIRECTORY_PATH / "results_new_ges"


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
# Scripts DYNAMIC STOCK
# ========================================

# === LOAD STOCK DATA ===
df_stock_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "NRJ" / "dwelling_stock_before_renovation.csv", sep=';', index_col=[0,1])
df_stock_col_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "NRJ" / "dwelling_stock_before_renovation_col.csv", sep=';', index_col=[0,1])
df_stock_ind_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "NRJ" / "dwelling_stock_before_renovation_ind.csv", sep=';', index_col=[0,1])

# === LOAD SCENARIOS ===
### Renovation and demolition pace in pyam format
renovation = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "renovation" / "renovation_scenarios.csv", sep=';')
demolition = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "demolition" / "demolition_scenarios.csv", sep=';')

### Operational and embodied GHGE ratios in kgCO2eq/m2
operational_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "energy_carriers_scenarios.csv", sep=';')
embodied_ghge_reno = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_renovation_scenarios.csv", sep=';')
embodied_ghge_dem = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_demolition_scenarios.csv", sep=';')

### List of available scenarios
print('renovation scenario: ', renovation.scenario)
print('demolition scenario: ', demolition.scenario)
print('operational GHGE scenario: ', operational_ghge.scenario)
print('embodied GHGE scenario for renovation: ', embodied_ghge_reno.scenario)
print('embodied GHGE scenario for demolition: ', embodied_ghge_dem.scenario)

### Pivot table from the High Renovation Scenarios
pt_col_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "collective_dwelling" / "pt_surface_col_hrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "individual_dwelling" / "pt_surface_ind_hrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the Medium Renovation Scenarios
pt_col_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "collective_dwelling" / "pt_surface_col_mrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "individual_dwelling" / "pt_surface_ind_mrs.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the High Demolition Scenarios
pt_col_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ"/"collective_dwelling" / "pt_surface_col_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"individual_dwelling" / "pt_surface_ind_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
### Pivot table from the Medium Demolition Scenarios
pt_col_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"collective_dwelling" / "pt_surface_col_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"individual_dwelling" / "pt_surface_ind_hds.csv", sep=';', index_col=[0,1], header=[0,1], skipinitialspace=True)


# === RUN SIMULATION YEAR BY YEAR AND UPDATE THE RELEVANT PARTS ===
## High Renovation Scenario and Middle Demolition Scenario
### Collective dwellings
updated_df_stock_hrs_linear_mds_col, changes_reno_lists_hrs_linear_col, changes_dem_lists_mds_col, delta_hrs_linear_mds_col = run_dynamic_simulation_nrj(
    pt_col_reno_hrs, pt_col_dem_mds, df_stock_col_2020,
    renovation_scenario='HRS_Linear', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_hrs_plateau_mds_col, changes_reno_lists_hrs_plateau_col, changes_dem_lists_mds_col, delta_hrs_plateau_mds_col = run_dynamic_simulation_nrj(
    pt_col_reno_hrs, pt_col_dem_mds, df_stock_col_2020,
    renovation_scenario='HRS_Plateau', renovation_variable='Collective_dwelling',
    demolition_scenario='MDS', demolition_variable='Collective_dwelling'
)

### Individual dwellings
updated_df_stock_hrs_linear_mds_ind, changes_reno_lists_hrs_linear_ind, changes_dem_lists_mds_ind, delta_hrs_linear_mds_ind = run_dynamic_simulation_nrj(
    pt_ind_reno_hrs, pt_ind_dem_mds, df_stock_ind_2020,
    renovation_scenario='HRS_Linear', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_hrs_plateau_mds_ind, changes_reno_lists_hrs_plateau_ind, changes_dem_lists_mds_ind, delta_hrs_plateau_mds_ind = run_dynamic_simulation_nrj(
    pt_ind_reno_hrs, pt_ind_dem_mds, df_stock_ind_2020,
    renovation_scenario='HRS_Plateau', renovation_variable='Individual_dwelling',
    demolition_scenario='MDS', demolition_variable='Individual_dwelling'
)


### Middle Renovation Scenario and High Demolition Scenario
### Collective dwellings
updated_df_stock_mrs_linear_hds_col, changes_reno_lists_mrs_linear_col, changes_dem_lists_hds_col, delta_mrs_linear_hds_col = run_dynamic_simulation_nrj(
    pt_col_reno_mrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_Linear', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

updated_df_stock_mrs_plateau_hds_col, changes_reno_lists_mrs_plateau_col, changes_dem_lists_hds_col, delta_mrs_plateau_hds_col = run_dynamic_simulation_nrj(
    pt_col_reno_mrs, pt_col_dem_hds, df_stock_col_2020,
    renovation_scenario='MRS_Plateau', renovation_variable='Collective_dwelling',
    demolition_scenario='HDS', demolition_variable='Collective_dwelling'
)

### Individual dwellings
updated_df_stock_mrs_linear_hds_ind, changes_reno_lists_mrs_linear_ind, changes_dem_lists_hds_ind, delta_mrs_linear_hds_ind = run_dynamic_simulation_nrj(
    pt_ind_reno_mrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_Linear', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

updated_df_stock_mrs_plateau_hds_ind, changes_reno_lists_mrs_plateau_ind, changes_dem_lists_hds_ind, delta_mrs_plateau_hds_ind = run_dynamic_simulation_nrj(
    pt_ind_reno_mrs, pt_ind_dem_hds, df_stock_ind_2020,
    renovation_scenario='MRS_Plateau', renovation_variable='Individual_dwelling',
    demolition_scenario='HDS', demolition_variable='Individual_dwelling'
)

# === GROUP RENOVATION BY JUMPS ===
## Collective dwelling
df_jumps_reno_hrs_linear_col = calculate_jumps_epc_label(changes_reno_lists_hrs_linear_col)
df_jumps_reno_hrs_plateau_col = calculate_jumps_epc_label(changes_reno_lists_hrs_plateau_col)
df_jumps_reno_mrs_linear_col = calculate_jumps_epc_label(changes_reno_lists_mrs_linear_col)
df_jumps_reno_mrs_plateau_col = calculate_jumps_epc_label(changes_reno_lists_mrs_plateau_col)

iamdf_reno_hrs_linear_col = create_iamdf_reno(df_jumps_reno_hrs_linear_col,
                                   model_name='CSTB',
                                   scenario_name='HRS_linear',
                                   variable_name='Collective_dwelling')

iamdf_reno_hrs_plateau_col = create_iamdf_reno(df_jumps_reno_hrs_plateau_col,
                                   model_name='CSTB',
                                   scenario_name='HRS_plateau',
                                   variable_name='Collective_dwelling')

iamdf_reno_mrs_linear_col = create_iamdf_reno(df_jumps_reno_mrs_linear_col,
                                   model_name='CSTB',
                                   scenario_name='MRS_linear',
                                   variable_name='Collective_dwelling')

iamdf_reno_mrs_plateau_col = create_iamdf_reno(df_jumps_reno_mrs_plateau_col,
                                   model_name='CSTB',
                                   scenario_name='MRS_plateau',
                                   variable_name='Collective_dwelling')

combined_dfs_reno_col = pd.concat([iamdf_reno_hrs_linear_col.data, iamdf_reno_hrs_plateau_col.data, iamdf_reno_mrs_linear_col.data, iamdf_reno_mrs_plateau_col.data], ignore_index=True)
combined_iamdfs_reno_col = pyam.IamDataFrame(combined_dfs_reno_col)

## Individual dwelling
df_jumps_reno_hrs_linear_ind = calculate_jumps_epc_label(changes_reno_lists_hrs_linear_ind)
df_jumps_reno_hrs_plateau_ind = calculate_jumps_epc_label(changes_reno_lists_hrs_plateau_ind)
df_jumps_reno_mrs_linear_ind = calculate_jumps_epc_label(changes_reno_lists_mrs_linear_ind)
df_jumps_reno_mrs_plateau_ind = calculate_jumps_epc_label(changes_reno_lists_mrs_plateau_ind)

iamdf_reno_hrs_linear_ind = create_iamdf_reno(df_jumps_reno_hrs_linear_ind,
                                   model_name='CSTB',
                                   scenario_name='HRS_linear',
                                   variable_name='Individual_dwelling')

iamdf_reno_hrs_plateau_ind = create_iamdf_reno(df_jumps_reno_hrs_plateau_ind,
                                   model_name='CSTB',
                                   scenario_name='HRS_plateau',
                                   variable_name='Individual_dwelling')

iamdf_reno_mrs_linear_ind = create_iamdf_reno(df_jumps_reno_mrs_linear_ind,
                                   model_name='CSTB',
                                   scenario_name='MRS_linear',
                                   variable_name='Individual_dwelling')

iamdf_reno_mrs_plateau_ind = create_iamdf_reno(df_jumps_reno_mrs_plateau_ind,
                                   model_name='CSTB',
                                   scenario_name='MRS_plateau',
                                   variable_name='Individual_dwelling')

combined_dfs_reno_ind = pd.concat([iamdf_reno_hrs_linear_ind.data, iamdf_reno_hrs_plateau_ind.data, iamdf_reno_mrs_linear_ind.data, iamdf_reno_mrs_plateau_ind.data], ignore_index=True)
combined_iamdfs_reno_ind = pyam.IamDataFrame(combined_dfs_reno_ind)

# Residential dwelling
combined_dfs_reno = pd.concat([combined_iamdfs_reno_col.data, combined_iamdfs_reno_ind.data], ignore_index=True)
combined_iamdfs_reno = pyam.IamDataFrame(combined_dfs_reno)
combined_iamdfs_reno_res = add_residential_dwelling_reno(combined_iamdfs_reno)
combined_iamdfs_reno_res = group_saut_dpe_reno(combined_iamdfs_reno)
combined_iamdfs_reno_res.to_csv(RESULT_NEW_NRJ_PATH / "combined_reno_surface_iamdf.csv", sep=';')

# == CREATE IAMDF STOCK ==
## HRS with MDS
### Collective dwelling
iamdf_stock_hrs_linear_mds_col = create_iamdf_stock(
                        updated_df_stock_hrs_linear_mds_col,
                         scenario_name='HRS_linear_MDS',
                         variable_name='Collective_dwelling')

iamdf_stock_hrs_plateau_mds_col = create_iamdf_stock(
                        updated_df_stock_hrs_plateau_mds_col,
                         scenario_name='HRS_plateau_MDS',
                         variable_name='Collective_dwelling')

### Individual dwelling
iamdf_stock_hrs_linear_mds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_linear_mds_ind,
                         scenario_name='HRS_linear_MDS',
                         variable_name='Individual_dwelling')

iamdf_stock_hrs_plateau_mds_ind = create_iamdf_stock(
                        updated_df_stock_hrs_plateau_mds_ind,
                         scenario_name='HRS_plateau_MDS',
                         variable_name='Individual_dwelling')

## MRS with HDS
### Collective dwelling
iamdf_stock_mrs_linear_hds_col = create_iamdf_stock(
                        updated_df_stock_mrs_linear_hds_col,
                         scenario_name='MRS_linear_HDS',
                         variable_name='Collective_dwelling')

iamdf_stock_mrs_plateau_hds_col = create_iamdf_stock(
                        updated_df_stock_mrs_plateau_hds_col,
                         scenario_name='MRS_plateau_HDS',
                         variable_name='Collective_dwelling')

### Individual dwelling
iamdf_stock_mrs_linear_hds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_linear_hds_ind,
                         scenario_name='MRS_linear_HDS',
                         variable_name='Individual_dwelling')

iamdf_stock_mrs_plateau_hds_ind = create_iamdf_stock(
                        updated_df_stock_mrs_plateau_hds_ind,
                         scenario_name='MRS_plateau_HDS',
                         variable_name='Individual_dwelling')

## All together
combined_iamdfs_stock = pd.concat([iamdf_stock_hrs_linear_mds_col.data,
                                   iamdf_stock_hrs_plateau_mds_col.data,
                                   iamdf_stock_hrs_linear_mds_ind.data,
                                   iamdf_stock_hrs_plateau_mds_ind.data,
                                   iamdf_stock_mrs_linear_hds_col.data,
                                   iamdf_stock_mrs_plateau_hds_col.data,
                                   iamdf_stock_mrs_linear_hds_ind.data,
                                   iamdf_stock_mrs_plateau_hds_ind.data], ignore_index=True)
combined_iamdfs_stock = pyam.IamDataFrame(combined_iamdfs_stock)

### We add the number of residential dwelling
combined_iamdfs_stock_res = add_residential_dwelling_stock(combined_iamdfs_stock)
#combined_iamdfs_stock_res.to_csv(RESULT_NEW_NRJ_PATH / "combined_stock_nrj_iamdf.csv", sep=';')

# ========================================
# Scripts NRJ
# ========================================
combined_construction_operational_cumulated_iamdf = pyam.IamDataFrame((RESULT_NEW_NRJ_PATH / "combined_construction_nrj_cumulated_iamdf.csv"), sep=';')

combined_all_stock_nrj_iamdf = operational_all_stock_combined(combined_construction_operational_cumulated_iamdf, combined_iamdfs_stock_res)
combined_all_stock_nrj_iamdf = drop_inconsistent_nrj_scenarios(combined_all_stock_nrj_iamdf)
combined_all_stock_nrj_iamdf.to_csv(RESULT_NEW_NRJ_PATH / "combined_all_stock_nrj_iamdf.csv", sep=';')

# ========================================
# Scripts GHGE
# ========================================

# == CALCULATE EMBODIED GHGE OF RENOVATION ==
combined_reno_embodied_iamdf = calculate_combined_embodied_reno_iamdf(combined_iamdfs_reno, embodied_ghge_reno)
combined_reno_embodied_iamdf = group_saut_dpe_reno(combined_reno_embodied_iamdf)
combined_reno_embodied_iamdf.to_csv(RESULT_NEW_NRJ_PATH / "combined_reno_embodied_iamdf.csv", sep=';')


# == CALCULATE EMBODIED GHGE OF DEMOLITION ==
combined_demolition_embodied_iamdf = calculate_combined_ghge_new_construction(demolition, embodied_ghge_dem)
combined_demolition_embodied_iamdf.to_csv(RESULT_NEW_NRJ_PATH / "combined_demolition_embodied_iamdf.csv", sep=';')


# == CALCULATE OPERATIONAL GHGE OF THE STOCK AFTER RENOVATION AND DEMOLITION ==
combined_iamdfs_stock_operational = calculate_combined_operational_stock_iamdf(combined_iamdfs_stock_res, operational_ghge)
combined_iamdfs_stock_operational.to_csv(RESULT_NEW_NRJ_PATH / "combined_stock_operational_iamdf.csv", sep=';')


# == COMBINE EMBODIED GHGE OF ALL THE STOCK, E.G. NEW CONSTRUCTION, RENOVATION & DEMOLITION ==
combined_construction_embodied_iamdf = pyam.IamDataFrame((RESULT_NEW_NRJ_PATH / "combined_construction_embodied_iamdf.csv"), sep=';')

## For one particular scenario
embodied_bau = embodied_all_stock(
                            combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf,
                            created_scenario_name='STEPS',
                            construction_scenario='BAU_construction_BAU_RE2020_upfront',
                            renovation_scenario='MRS_linear_BAU_RE2020_WLC',
                            demolition_scenario='HDS_Constant_WLC',
                            construction_variable='Residential_dwelling',
                            renovation_variable='Residential_dwelling',
                            demolition_variable='Residential_dwelling')

embodied_s2 = embodied_all_stock(
                            combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf,
                            created_scenario_name='APS',
                            construction_scenario='S2_construction_Optimist_RE2020_upfront',
                            renovation_scenario='HRS_plateau_Optimist_RE2020_WLC',
                            demolition_scenario='MDS_Constant_WLC',
                            construction_variable='Residential_dwelling',
                            renovation_variable='Residential_dwelling',
                            demolition_variable='Residential_dwelling')

## For all scenarios combination
combined_embodied_iamdf = embodied_all_stock_combined(combined_construction_embodied_iamdf, combined_reno_embodied_iamdf, combined_demolition_embodied_iamdf)
combined_embodied_iamdf = drop_inconsistent_embodied_scenarios(combined_embodied_iamdf)
combined_embodied_iamdf.to_csv(RESULT_NEW_NRJ_PATH / "combined_all_stock_embodied_iamdf.csv", sep=';')


# == COMBINE OPERATIONAL GHGE OF ALL THE STOCK, E.G. NEW CONSTRUCTION, RENOVATION & DEMOLITION ==
combined_construction_operational_iamdf = pyam.IamDataFrame((RESULT_NEW_NRJ_PATH / "combined_construction_operational_cumulated_iamdf.csv"), sep=';')

# For one particular scenario
operational_bau = operational_all_stock(
                            combined_construction_operational_iamdf, combined_iamdfs_stock_operational,
                            construction_scenario='BAU_construction_Constant',
                            remaining_scenario='MRS_linear_HDS_Constant',
                            created_scenario_name='STEPS')

operational_bau.to_csv(RESULT_NEW_NRJ_PATH / "operational_bau.csv", sep=';')

operational_s2 = operational_all_stock(
                            combined_construction_operational_iamdf, combined_iamdfs_stock_operational,
                            construction_scenario='S2_construction_Quarter',
                            remaining_scenario='HRS_plateau_MDS_Quarter',
                            created_scenario_name='APS')

# For all scenarios combination
combined_operational_iamdf = operational_all_stock_combined(combined_construction_operational_iamdf, combined_iamdfs_stock_operational)
combined_operational_iamdf = drop_inconsistent_operational_scenarios(combined_operational_iamdf)
combined_operational_iamdf.to_csv(RESULT_NEW_NRJ_PATH / "combined_all_stock_operational_iamdf.csv", sep=';')


# ========================================
# GRAPHS GHGE
# ========================================

# # == EMBODIED GHGE ==
# ## ANNUAL
# fig_embodied_bau = plot_embodied_all_stock(embodied_bau,
#                       title="Embodied GHGE - STEPS")
#
# fig_embodied_s2 = plot_embodied_all_stock(embodied_s2,
#                       title="Embodied GHGE - APS")
#
# ## CUMULATIVE
# fig_embodied_bau_cum = plot_cumulative_all_stock_embodied(embodied_bau,
#                       title="Cumulated embodied GHGE - STEPS")
#
# fig_embodied_s2_cum = plot_cumulative_all_stock_embodied(embodied_s2,
#                       title="Cumulated embodied GHGE - APS")
#
#
# # == OPERATIONAL GHGE ==
# ## ANNUAL
fig_operational_bau_dpe = plot_operational_all_stock_dpe(operational_bau,
                                                     scenario_pattern='STEPS',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - STEPS',
                                                     figsize=(10,8))

fig_operational_s2_dpe = plot_operational_all_stock_dpe(operational_s2,
                                                     scenario_pattern='APS',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - APS',
                                                     figsize=(10,8))
#
fig_operational_bau_nrj = plot_operational_all_stock_nrj(operational_bau,
                                                     scenario_pattern='STEPS',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - STEPS - EPC NRJ',
                                                     figsize=(10,8))

fig_operational_s2_nrj = plot_operational_all_stock_nrj(operational_s2,
                                                     scenario_pattern='APS',
                                                     variable='Residential_dwelling',
                                                     title='Operational GHGE - APS - EPC NRJ',
                                                     figsize=(10,8))

#
# ## CUMULATIVE
fig_operational_bau_dpe_cum = plot_cumulative_operational_all_stock_dpe(operational_bau,
                                                                scenario_pattern='STEPS',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - STEPS - EPC NRJ',
                                                                figsize=(10,8))


fig_operational_s2_dpe_cum = plot_cumulative_operational_all_stock_dpe(operational_s2,
                                                                scenario_pattern='APS',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - APS - EPC NRJ',
                                                                figsize=(10,8))

fig_operational_bau_nrj_cum = plot_cumulative_operational_all_stock_nrj(operational_bau,
                                                                scenario_pattern='STEPS',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - STEPS - EPC NRJ',
                                                                figsize=(10,8))


fig_operational_s2_nrj_cum = plot_cumulative_operational_all_stock_nrj(operational_s2,
                                                                scenario_pattern='APS',
                                                                variable='Residential_dwelling',
                                                                title='Cumulated operational GHGE - APS - EPC NRJ',
                                                                figsize=(10,8))


# # == WLC emissions ==
# ## Annual
# wlc_2050 = wlc_graph_compare_scenarios(operational_bau, operational_s2,
#                                        embodied_bau, embodied_s2,
#                                        scenario_name_1='STEPS', scenario_name_2='APS',
#                                        variable='Residential_dwelling',
#                                        year=2050)
#
#
# ## Budgets
#
# # ========================================
# # GRAPHS NRJ
# # ========================================
fig_bau_nrj = plot_stock_nrj_twh(combined_all_stock_nrj_iamdf,
                                                                scenario_pattern='BAU_construction_MRS_linear_HDS',
                                                                variable='Residential_dwelling',
                                                                title='TWh - STEPS - EPC NRJ',
                                                                figsize=(10,8))

fig_s2_nrj = plot_stock_nrj_twh(combined_all_stock_nrj_iamdf,
                                                                scenario_pattern='S2_construction_HRS_plateau_MDS',
                                                                variable='Residential_dwelling',
                                                                title='TWh - APS - EPC NRJ',
                                                                figsize=(10,8))

# ========================================
# SENSIBILITY GRAPHS
# ========================================
# Operational range of values
operational_steps_range = plot_range_operational_ghge(combined_operational_iamdf,
                                                         variable='Residential_dwelling',
                                                        scenario='BAU*',
                                                         title='Range of operational GHGE values for STEPS')

operational_aps_range = plot_range_operational_ghge(combined_operational_iamdf,
                                                         variable='Residential_dwelling',
                                                        scenario='S2*',
                                                         title='Range of operational GHGE values for APS')


# Embodied range of values
embodied_steps_range = plot_range_embodied_ghge(combined_embodied_iamdf,
                                                         variable='Residential_dwelling',
                                                        scenario='BAU*',
                                                         title='Range of embodied GHGE values for STEPS')

embodied_aps_range = plot_range_embodied_ghge(combined_embodied_iamdf,
                                                         variable='Residential_dwelling',
                                                        scenario='S2*',
                                                         title='Range of embodied GHGE values for APS')