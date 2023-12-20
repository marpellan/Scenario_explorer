# -*- coding: utf-8 -*-

# ========================================
# External imports
# ========================================

import pandas as pd
import pyam

from collections import defaultdict
from importlib import resources

# ========================================
# External CSTB imports
# ========================================


# ========================================
# Internal imports
# ========================================

from trajectoire.config import data

# ========================================
# Constants
# ========================================

DATA_DIRECTORY_PATH = resources.files(data)
SCENARIOS_DIRECTORY_PATH = DATA_DIRECTORY_PATH / "scenarios"
CALIBRATION_DIRECTORY_PATH = DATA_DIRECTORY_PATH / "calibration"


MODEL_DIRECTORY_PATH = DATA_DIRECTORY_PATH / "calibration"
folder = r'C:\Users\pellan\OneDrive - CSTBGroup\Thèse stratégie carbone\Articles\Journal_papers\2nd article\Gitlab'
model_folder = folder + '\MODEL'
results_folder = r'C:\Users\pellan\OneDrive - CSTBGroup\Thèse stratégie carbone\Articles\Journal_papers\2nd article\Gitlab\MODEL\results\results_epc_matrix'

# ========================================
# Variables
# ========================================

# === LOAD STOCK DATA ===
df_stock_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "dwelling_stock_before_renovation.csv", sep=";", index_col=[0,1])

# === LOAD SCENARIOS ===
### Renovation pace in pyam format
renovation = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "renovation" / "renovation_scenarios_bdnb.csv", sep=";")
demolition = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "demolition" / "demolition_scenarios_bdnb.csv", sep=";")

### Pivot table from the High Renovation Scenarios
pt_col_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_hrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_hrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the Medium Renovation Scenarios
pt_col_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_mrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_mrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the High Demolition Scenarios
pt_col_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "collective_dwelling" / "pt_surface_col_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "individual_dwelling" / "pt_surface_ind_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the Medium Demolition Scenarios
pt_col_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios\collective_dwelling\pt_surface_col_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios\individual_dwelling\pt_surface_ind_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

# ========================================
# Classes
# ========================================


# ========================================
# Functions
# ========================================

def update_pivot_table_reno(pivot_table_reno, target_number, start_row=6):
    if not isinstance(pivot_table_reno, pd.DataFrame) or not isinstance(pivot_table_reno.index, pd.MultiIndex):
        raise ValueError("Input must be a DataFrame with a MultiIndex.")

    # Create a copy of the DataFrame to perform updates
    updated_df = pivot_table_reno.copy()

    total = 0
    changes = []  # List to store changes made
    changes_df = pd.DataFrame(0, index=pivot_table_reno.index, columns=pivot_table_reno.index)

    for i in range(start_row, -1, -1):
        for j in range(len(updated_df.columns)):
            index = updated_df.index[i]
            column = updated_df.columns[j]
            value = updated_df.loc[index, column]

            if total + value <= target_number:
                total += value
                changes.append((index, column, value))  # Store change information
                changes_df.loc[index, column] = value  # Update changes_df with value
                updated_df.loc[index, column] = 0
                if total == target_number:
                    return updated_df, changes_df, changes
            else:
                diff = target_number - total
                removed_amount = value - diff
                changes.append((index, column, diff))  # Store change information
                changes_df.loc[index, column] = diff  # Update changes_df with diff
                updated_df.loc[index, column] -= diff
                total = target_number
                return updated_df, changes_df, changes

    return updated_df, changes_df, changes


def update_pivot_table_dem(pivot_table_dem, target_number, start_row=6):
    if not isinstance(pivot_table_dem, pd.DataFrame) or not isinstance(pivot_table_dem.index, pd.MultiIndex):
        raise ValueError("Input must be a DataFrame with a MultiIndex.")

    # Create a copy of the DataFrame to perform updates
    updated_df = pivot_table_dem.copy()

    total = 0
    changes = []  # List to store changes made
    changes_df = pd.DataFrame(0, index=pivot_table_dem.index, columns=pivot_table_dem.index)

    for i in range(start_row, -1, -1):
        for j in range(start_row, -1, -1):
            index = updated_df.index[i]
            column = updated_df.columns[j]
            value = updated_df.loc[index, column]

            if total + value <= target_number:
                total += value
                changes.append((index, column, value))  # Store change information
                changes_df.loc[index, column] = value  # Update changes_df with value
                updated_df.loc[index, column] = 0
                if total == target_number:
                    return updated_df, changes_df, changes
            else:
                diff = target_number - total
                removed_amount = value - diff
                changes.append((index, column, diff))  # Store change information
                changes_df.loc[index, column] = diff  # Update changes_df with diff
                updated_df.loc[index, column] -= diff
                total = target_number
                return updated_df, changes_df, changes

    return updated_df, changes_df, changes


def update_surface_df(surface_df, changes_renovation, changes_demolition):
    updated_surface_df = surface_df.copy()

    for change in changes_renovation:
        from_index = change[0]
        to_index = change[1]
        amount = change[2]

        updated_surface_df.loc[from_index] -= amount
        updated_surface_df.loc[to_index] += amount

    for change in changes_demolition:
        from_index = change[0]
        to_index = change[1]
        amount = change[2]

        updated_surface_df.loc[from_index] -= amount

    changes_stock_df = updated_surface_df - surface_df

    return updated_surface_df


def run_dynamic_simulation(pivot_table_reno, pivot_table_dem, surface_df,
                           renovation_scenario=None, renovation_variable=None,
                           demolition_scenario=None, demolition_variable=None):
    updated_reno_dfs = {}  # To store updated pivot tables for renovation for each year
    updated_dem_dfs = {}  # To store updated pivot tables for demolition for each year
    updated_surface_dfs = {}  # To store updated stock dataFrames for each year

    changes_reno_lists = {}  # To store targeted segments renovated for each year, in a list format
    changes_dem_lists = {}  # To store targeted segments demolished for each year, in a list format

    changes_reno_dfs = {}  # To store targeted segments renovated for each year, in a list format
    changes_dem_dfs = {}  # To store targeted segments demolished for each year, in a list format

    for year in range(2020, 2051):
        target_number_reno = \
        renovation.filter(year=year, scenario=renovation_scenario, variable=renovation_variable, unit='Surface in m2').data[
            'value'].values[0]
        target_number_dem = \
        demolition.filter(year=year, scenario=demolition_scenario, variable=demolition_variable, unit='Surface in m2').data[
            'value'].values[0]

        updated_reno_df, changes_reno_df, changes_reno = update_pivot_table_reno(pivot_table_reno, target_number_reno)
        updated_dem_df, changes_dem_df, changes_dem = update_pivot_table_dem(pivot_table_dem, target_number_dem)
        updated_surface_df = update_surface_df(surface_df, changes_reno, changes_dem)

        updated_reno_dfs[year + 1] = updated_reno_df
        updated_dem_dfs[year + 1] = updated_dem_df
        updated_surface_dfs[year + 1] = updated_surface_df

        changes_reno_lists[year] = changes_reno
        changes_dem_lists[year] = changes_dem

        changes_reno_dfs[year] = changes_reno_df
        changes_dem_dfs[year] = changes_dem_df

        # Use updated data for the next year's calculations
        pivot_table_reno = updated_reno_df
        pivot_table_dem = updated_dem_df
        surface_df = updated_surface_df

    return (updated_surface_dfs, changes_reno_lists, changes_dem_lists)


def calculate_jumps_epc_label(data):
    max_jumps = 6
    jumps_data = defaultdict(lambda: defaultdict(int))

    for year, entries in data.items():
        from_letter = None
        jumps_dict = defaultdict(int)

        for entry in entries:
            from_letter = entry[0][1]
            to_letter = entry[1][1]
            value = entry[2]

            if from_letter != to_letter:
                jumps = abs(ord(to_letter) - ord(from_letter))
                jumps_dict[jumps] += value

        for i in range(1, max_jumps + 1):
            jumps_data[year][i] = jumps_dict[i]

    jumps_df = pd.DataFrame(jumps_data)

    return jumps_df


def create_iamdf_reno(jumps_df, model_name, scenario_name, variable_name):
    jumps_df.index.name = 'saut_classe_dpe'

    iamdf = pyam.IamDataFrame(jumps_df,
                              model=model_name,
                              scenario=scenario_name,
                              region='France',
                              variable=variable_name,
                              unit='Surface in m2')

    return iamdf


def create_iamdf_stock(updated_surface_dfs,
                        scenario_name,
                        variable_name):
    summed_dfs = {}

    for year, updated_surface_df in updated_surface_dfs.items():
        # Transpose the DataFrame and store it in the dictionary
        summed_dfs[year] = updated_surface_df

    # Concatenate the transposed DataFrames to create the summary DataFrame
    summary_df = pd.concat(summed_dfs, axis=1)

    df = summary_df.droplevel(level=1, axis=1).droplevel(level=0, axis=0)

    iamdf = pyam.IamDataFrame(df,
                              model='CSTB',
                              scenario=scenario_name,
                              region='France',
                              variable=variable_name,
                              unit='Surface in m2')
    return iamdf


def add_residential_dwelling_reno(iamdf):
    # Make a copy of the dataframe
    iamdf_new = iamdf.copy().data

    # Filter rows for 'Collective_dwelling' and 'Individual_dwelling'
    collective_individual_df = iamdf_new[iamdf_new['variable'].isin(['Collective_dwelling', 'Individual_dwelling'])]

    # Group by 'scenario', 'year', 'model', 'region', and 'unit' and calculate the residential_dwelling_value for each group
    group_sum = collective_individual_df.groupby(['scenario', 'year', 'model', 'region', 'unit', 'saut_classe_dpe'])['value'].sum()

    # Convert the group_sum Series to a DataFrame
    group_sum_df = group_sum.reset_index()

    # Add 'Residential_dwelling' as the variable for each group
    group_sum_df['variable'] = 'Residential_dwelling'

    # Append the group_sum_df to the original dataframe
    iamdf_new = pd.concat(
        [iamdf_new, group_sum_df[['scenario', 'year', 'model', 'region', 'unit', 'variable', 'saut_classe_dpe', 'value']]],
        ignore_index=True)

    iamdf_new_output = pyam.IamDataFrame(iamdf_new)

    return iamdf_new_output


def group_saut_dpe_reno(iamdf):
    # Make a copy of the dataframe
    iamdf_new = iamdf.copy().data

    # Filter rows for 'Collective_dwelling' and 'Individual_dwelling'
    sum_df = iamdf_new[iamdf_new['saut_classe_dpe'].isin([1,2,3,4,5,6])]

    # Group by 'scenario', 'year', 'model', 'region', and 'unit' and calculate the residential_dwelling_value for each group
    group_sum = sum_df.groupby(['scenario', 'year', 'model', 'region', 'unit', 'variable'])['value'].sum()

    # Convert the group_sum Series to a DataFrame
    group_sum_df = group_sum.reset_index()

    # Add 'Residential_dwelling' as the variable for each group
    group_sum_df['saut_classe_dpe'] = 'sum'

    # Append the group_sum_df to the original dataframe
    iamdf_new = pd.concat(
        [iamdf_new, group_sum_df[['scenario', 'year', 'model', 'region', 'unit', 'variable', 'saut_classe_dpe', 'value']]],
        ignore_index=True)

    iamdf_new_output = pyam.IamDataFrame(iamdf_new)

    return iamdf_new_output


def add_residential_dwelling_stock(iamdf):
    # Make a copy of the dataframe
    iamdf_new = iamdf.copy().data

    # Filter rows for 'Collective_dwelling' and 'Individual_dwelling'
    collective_individual_df = iamdf_new[iamdf_new['variable'].isin(['Collective_dwelling', 'Individual_dwelling'])]

    # Group by 'scenario', 'year', 'model', 'region', and 'unit' and calculate the residential_dwelling_value for each group
    group_sum = collective_individual_df.groupby(['scenario', 'year', 'model', 'region', 'unit', 'dpe_avant'])['value'].sum()

    # Convert the group_sum Series to a DataFrame
    group_sum_df = group_sum.reset_index()

    # Add 'Residential_dwelling' as the variable for each group
    group_sum_df['variable'] = 'Residential_dwelling'

    # Append the group_sum_df to the original dataframe
    iamdf_new = pd.concat(
        [iamdf_new, group_sum_df[['scenario', 'year', 'model', 'region', 'unit', 'variable', 'dpe_avant', 'value']]],
        ignore_index=True)

    iamdf_new_output = pyam.IamDataFrame(iamdf_new)

    return iamdf_new_output


def calculate_combined_embodied_reno_iamdf(reno, embodied_ghge):
    '''
    Calculate embodied GHGE of renovations from the combination of surface and embodied GHGE per m2 scenarios
    Returns a single IamDataFrame with the values in MtCO2eq for both collective and individual dwellings
    '''

    reno_embodied_col_list = []
    reno_embodied_ind_list = []
    reno_embodied_res_list = []

    for reno_scenario in reno.scenario:
        reno_col_df = reno.filter(scenario=reno_scenario, variable='Collective_dwelling').data
        reno_ind_df = reno.filter(scenario=reno_scenario, variable='Individual_dwelling').data

        for scenario in embodied_ghge.scenario:
            embodied_ghge_col_df = embodied_ghge.filter(scenario=scenario, variable='Collective_dwelling').data
            embodied_ghge_ind_df = embodied_ghge.filter(scenario=scenario, variable='Individual_dwelling').data

            # Multiply m2 and kgCO2eq/m2
            reno_embodied_col = reno_col_df['value'] * embodied_ghge_col_df['value']/10e8
            reno_embodied_ind = reno_ind_df['value'] * embodied_ghge_ind_df['value']/10e8
            reno_embodied_res = reno_embodied_col + reno_embodied_ind

            scenario_name = f'{reno_scenario}_{scenario}'

            # Create a new dataframe with year as a column, scenarios name being the combination of reno and embodied_ghge scenarios
            result_col_df = pd.DataFrame(
                {'year': reno_col_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': reno_col_df.variable[0],
                 'unit': 'MtCO2eq',
                 'saut_classe_dpe': '',
                 'value': reno_embodied_col})

            result_ind_df = pd.DataFrame(
                {'year': reno_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': reno_ind_df.variable[0],
                 'unit': 'MtCO2eq',
                 'saut_classe_dpe': '',
                 'value': reno_embodied_ind})

            result_res_df = pd.DataFrame(
                {'year': reno_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': 'Residential_dwelling',
                 'unit': 'MtCO2eq',
                 'saut_classe_dpe': '',
                 'value': reno_embodied_res})

            reno_embodied_col_list.append(result_col_df)
            reno_embodied_ind_list.append(result_ind_df)
            reno_embodied_res_list.append(result_res_df)

    combined_reno_embodied_col_df = pd.concat(reno_embodied_col_list)
    combined_reno_embodied_ind_df = pd.concat(reno_embodied_ind_list)
    combined_reno_embodied_res_df = pd.concat(reno_embodied_res_list)

    # Add the 'saut_classe_dpe' column to both DataFrames
    combined_reno_embodied_col_df['saut_classe_dpe'] = (combined_reno_embodied_col_df.index % 6) + 1
    combined_reno_embodied_ind_df['saut_classe_dpe'] = (combined_reno_embodied_ind_df.index % 6) + 1
    combined_reno_embodied_res_df['saut_classe_dpe'] = (combined_reno_embodied_res_df.index % 6) + 1

    # Concatenate both DataFrames into a single IamDataFrame
    combined_reno_embodied_df = pd.concat([combined_reno_embodied_col_df, combined_reno_embodied_ind_df, combined_reno_embodied_res_df])

    # Create a single IamDataFrame containing all data
    combined_reno_embodied_iamdf = pyam.IamDataFrame(combined_reno_embodied_df)

    return combined_reno_embodied_iamdf


def calculate_combined_operational_stock_iamdf(stock, operational_ghge):
    '''
    Calculate operational GHGE of the stock
    Returns a single IamDataFrame with the values in MtCO2eq for both collective and individual dwellings
    '''

    operational_col_list = []
    operational_ind_list = []
    operational_res_list = []

    for stock_scenario in stock.scenario:
        stock_col_df = stock.filter(scenario=stock_scenario, variable='Collective_dwelling').data
        stock_ind_df = stock.filter(scenario=stock_scenario, variable='Individual_dwelling').data

        for scenario in operational_ghge.scenario:
            operational_ghge_col_df = operational_ghge.filter(scenario=scenario, variable='Collective_dwelling').data
            operational_ghge_ind_df = operational_ghge.filter(scenario=scenario, variable='Individual_dwelling').data

            # Multiply m2 and kgCO2eq/m2
            operational_col = stock_col_df['value'] * operational_ghge_col_df['value']/10e8
            operational_ind = stock_ind_df['value'] * operational_ghge_ind_df['value']/10e8
            operational_res = operational_col + operational_ind

            scenario_name = f'{stock_scenario}_{scenario}'

            # Create a new dataframe with year as a column, scenarios name being the combination of reno and embodied_ghge scenarios
            result_col_df = pd.DataFrame(
                {'year': stock_col_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': stock_col_df.variable[0],
                 'unit': 'MtCO2eq',
                 'DPE': '',
                 'value': operational_col})

            result_ind_df = pd.DataFrame(
                {'year': stock_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': stock_ind_df.variable[0],
                 'unit': 'MtCO2eq',
                 'DPE': '',
                 'value': operational_ind})

            result_res_df = pd.DataFrame(
                {'year': stock_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': 'Residential_dwelling',
                 'unit': 'MtCO2eq',
                 'DPE': '',
                 'value': operational_res})

            operational_col_list.append(result_col_df)
            operational_ind_list.append(result_ind_df)
            operational_res_list.append(result_res_df)

    combined_operational_col_df = pd.concat(operational_col_list)
    combined_operational_ind_df = pd.concat(operational_ind_list)
    combined_operational_res_df = pd.concat(operational_res_list)

    # Add the 'DPE' column to both DataFrames
    combined_operational_col_df['DPE'] = (combined_operational_col_df.index % 7) + 1
    combined_operational_ind_df['DPE'] = (combined_operational_ind_df.index % 7) + 1
    combined_operational_res_df['DPE'] = (combined_operational_res_df.index % 7) + 1


    # Concatenate both DataFrames into a single IamDataFrame
    combined_operational_df = pd.concat([combined_operational_col_df, combined_operational_ind_df, combined_operational_res_df])

    # Create a single IamDataFrame containing all data
    combined_operational_iamdf = pyam.IamDataFrame(combined_operational_df)

    return combined_operational_iamdf


def group_dpe_stock(iamdf):
    # Make a copy of the dataframe
    iamdf_new = iamdf.copy().data

    # Filter rows for 'Collective_dwelling' and 'Individual_dwelling'
    sum_df = iamdf_new[iamdf_new['dpe'].isin([1,2,3,4,5,6,7])]

    # Group by 'scenario', 'year', 'model', 'region', and 'unit' and calculate the residential_dwelling_value for each group
    group_sum = sum_df.groupby(['scenario', 'year', 'model', 'region', 'unit', 'variable'])['value'].sum()

    # Convert the group_sum Series to a DataFrame
    group_sum_df = group_sum.reset_index()

    # Add 'Residential_dwelling' as the variable for each group
    group_sum_df['dpe'] = 'sum'

    # Append the group_sum_df to the original dataframe
    iamdf_new = pd.concat(
        [iamdf_new, group_sum_df[['scenario', 'year', 'model', 'region', 'unit', 'variable', 'dpe', 'value']]],
        ignore_index=True)

    iamdf_new_output = pyam.IamDataFrame(iamdf_new)

    return iamdf_new_output


def embodied_all_stock(construction, renovation, demolition,
                       construction_scenario=None, renovation_scenario=None, demolition_scenario=None,
                       construction_variable=None, renovation_variable=None, demolition_variable=None,
                       created_scenario_name=None):

    # Filter data from the three input dataframes based on provided criteria

    df_construction = construction.filter(scenario=construction_scenario, variable=construction_variable).data
    df_renovation = renovation.filter(scenario=renovation_scenario, variable=renovation_variable,
                                      saut_classe_dpe='sum').data
    df_demolition = demolition.filter(scenario=demolition_scenario, variable=demolition_variable).data

    # Create a new dataframe with year as column and an additional 'Programmation columns'
    construction_df = pd.DataFrame(
        {'year': df_construction['year'],
         'model': 'CSTB',
         'scenario': created_scenario_name,
         'region': 'France',
         'variable': df_construction.variable[0],
         'unit': 'MtCO2eq',
         'Programmation': 'New_construction',
         'value': df_construction['value']})

    renovation_df = pd.DataFrame(
        {'year': df_renovation['year'],
         'model': 'CSTB',
         'scenario': created_scenario_name,
         'region': 'France',
         'variable': df_renovation.variable[0],
         'unit': 'MtCO2eq',
         'Programmation': 'Renovation',
         'value': df_renovation['value']})

    demolition_df = pd.DataFrame(
        {'year': df_demolition['year'],
         'model': 'CSTB',
         'scenario': created_scenario_name,
         'region': 'France',
         'variable': df_demolition.variable[0],
         'unit': 'MtCO2eq',
         'Programmation': 'Demolition',
         'value': df_demolition['value']})

    # Concatenate the 3 dataframes together, since they have the same variables and scenario names
    combined_programmation_df = pd.concat([construction_df, renovation_df, demolition_df])

    # Create a single IamDataFramer containing all the data
    combined_programmation_iamdf = pyam.IamDataFrame(combined_programmation_df)

    return combined_programmation_iamdf


def operational_all_stock(construction, remaining,
                          construction_scenario=None, remaining_scenario=None,
                          created_scenario_name=None):
    df_construction = construction.filter(scenario=construction_scenario).data
    df_remaining = remaining.filter(scenario=remaining_scenario).data

    # We drop the sum in remaining
    df_remaining.drop(df_remaining.loc[df_remaining['dpe'] == 'sum'].index, inplace=True)

    # We add blabala
    df_construction['year'] = df_construction['year'] + 1
    df_construction['dpe'] = 'new'

    # Concatenate the 2 dataframes together, since they have the same variables and scenario names
    combined_programmation_df = pd.concat([df_construction, df_remaining])

    # We put the same scenario name
    combined_programmation_df['scenario'] = created_scenario_name

    # Create a single IamDataFramer containing all the data
    combined_programmation_iamdf = pyam.IamDataFrame(combined_programmation_df)

    return combined_programmation_iamdf


def embodied_all_stock_combined(construction, renovation, demolition):
    embodied_construction_col_list = []
    embodied_construction_ind_list = []
    embodied_construction_res_list = []
    embodied_renovation_col_list = []
    embodied_renovation_ind_list = []
    embodied_renovation_res_list = []
    embodied_demolition_col_list = []
    embodied_demolition_ind_list = []
    embodied_demolition_res_list = []

    for construction_scenario in construction.scenario:
        df_construction_col = construction.filter(scenario=construction_scenario, variable='Collective_dwelling').data
        df_construction_ind = construction.filter(scenario=construction_scenario, variable='Individual_dwelling').data

        for renovation_scenario in renovation.scenario:
            df_renovation_col = renovation.filter(scenario=renovation_scenario, variable='Collective_dwelling',
                                                  saut_classe_dpe='sum').data
            df_renovation_ind = renovation.filter(scenario=renovation_scenario, variable='Individual_dwelling',
                                                  saut_classe_dpe='sum').data

            for demolition_scenario in demolition.scenario:
                df_demolition_col = demolition.filter(scenario=demolition_scenario, variable='Collective_dwelling').data
                df_demolition_ind = demolition.filter(scenario=demolition_scenario, variable='Individual_dwelling').data

                created_scenario_name = f'{construction_scenario}_{renovation_scenario}_{demolition_scenario}'

                # Create a new dataframe with year as column and an additional 'Programmation columns'
                construction_df_col = pd.DataFrame(
                    {'year': df_construction_col['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Collective_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'New_construction',
                     'value': df_construction_col['value']})

                construction_df_ind = pd.DataFrame(
                    {'year': df_construction_ind['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Individual_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'New_construction',
                     'value': df_construction_ind['value']})

                construction_df_res = pd.DataFrame(
                    {'year': df_construction_col['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Residential_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'New_construction',
                     'value': df_construction_col['value'] + df_construction_ind['value']})

                renovation_df_col = pd.DataFrame(
                    {'year': df_renovation_col['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Collective_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Renovation',
                     'value': df_renovation_col['value']})

                renovation_df_ind = pd.DataFrame(
                    {'year': df_renovation_ind['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Individual_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Renovation',
                     'value': df_renovation_ind['value']})

                renovation_df_res = pd.DataFrame(
                    {'year': df_renovation_col['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Residential_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Renovation',
                     'value': df_renovation_col['value'] + df_renovation_ind['value']})

                demolition_df_col = pd.DataFrame(
                    {'year': df_demolition_col['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Collective_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Demolition',
                     'value': df_demolition_col['value']})

                demolition_df_ind = pd.DataFrame(
                    {'year': df_demolition_ind['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Individual_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Demolition',
                     'value': df_demolition_ind['value']})

                demolition_df_res = pd.DataFrame(
                    {'year': df_demolition_ind['year'],
                     'model': 'CSTB',
                     'scenario': created_scenario_name,
                     'region': 'France',
                     'variable': 'Residential_dwelling',
                     'unit': 'MtCO2eq',
                     'Programmation': 'Demolition',
                     'value': df_demolition_col['value'] + df_demolition_ind['value']})

                embodied_construction_col_list.append(construction_df_col)
                embodied_construction_ind_list.append(construction_df_ind)
                embodied_construction_res_list.append(construction_df_res)
                embodied_renovation_col_list.append(renovation_df_col)
                embodied_renovation_ind_list.append(renovation_df_ind)
                embodied_renovation_res_list.append(renovation_df_res)
                embodied_demolition_col_list.append(demolition_df_col)
                embodied_demolition_ind_list.append(demolition_df_ind)
                embodied_demolition_res_list.append(demolition_df_res)

    embodied_construction_col_df = pd.concat(embodied_construction_col_list)
    embodied_construction_ind_df = pd.concat(embodied_construction_ind_list)
    embodied_construction_res_df = pd.concat(embodied_construction_res_list)
    embodied_renovation_col_df = pd.concat(embodied_renovation_col_list)
    embodied_renovation_ind_df = pd.concat(embodied_renovation_ind_list)
    embodied_renovation_res_df = pd.concat(embodied_renovation_res_list)
    embodied_demolition_col_df = pd.concat(embodied_demolition_col_list)
    embodied_demolition_ind_df = pd.concat(embodied_demolition_ind_list)
    embodied_demolition_res_df = pd.concat(embodied_demolition_res_list)

    combined_embodied_df = pd.concat(
        [embodied_construction_col_df, embodied_construction_ind_df, embodied_construction_res_df,
         embodied_renovation_col_df, embodied_renovation_ind_df, embodied_renovation_res_df,
         embodied_demolition_col_df, embodied_demolition_ind_df, embodied_demolition_res_df])

    combined_embodied_iamdf = pyam.IamDataFrame(combined_embodied_df)

    return combined_embodied_iamdf


def operational_all_stock_combined(construction, remaining):
    combined_list = []

    for construction_scenario in construction.scenario:
        df_construction = construction.filter(scenario=construction_scenario).data

        # We add blabala
        df_construction['year'] = df_construction['year'] + 1
        df_construction['dpe'] = 'new'

        for remaining_scenario in remaining.scenario:
            df_remaining = remaining.filter(scenario=remaining_scenario).data

            # We drop the sum in remaining
            df_remaining.drop(df_remaining.loc[df_remaining['dpe'] == 'sum'].index, inplace=True)

            # We combine the two and put the scenario name
            combined_programmation_df = pd.concat([df_construction, df_remaining])
            created_scenario_name = f'{construction_scenario}_{remaining_scenario}'
            combined_programmation_df['scenario'] = created_scenario_name

            combined_list.append(combined_programmation_df)

    combined_df = pd.concat(combined_list)
    combined_iamdf = pyam.IamDataFrame(combined_df)

    return combined_iamdf


def drop_inconsistent_operational_scenarios(combined_operational_iamdf):
    # Define a dictionary of requirements for text0_text1 and text5
    requirement_dict = {
        'BAU_construction': 'HDS',
        'S2_construction': 'MDS',
        # Add more requirements as needed
    }

    # Create an empty list to store scenarios to keep
    scenarios_to_keep = []

    # Iterate through scenario names
    for scenario in combined_operational_iamdf.data['scenario']:
        # Split the scenario name into parts using underscores
        parts = scenario.split("_")

        # Check if the scenario name has at least 7 parts (text0_text1_text2_text3_text4_text5_text6)
        if len(parts) >= 7:
            # Check if text0_text1 is in the requirement_dict
            if f'{parts[0]}_{parts[1]}' in requirement_dict:
                # Check if text5 matches the requirement for text0_text1
                if parts[5] == requirement_dict[f'{parts[0]}_{parts[1]}']:
                    # Check if text2 is equal to text6
                    if parts[2] == parts[6]:
                        scenarios_to_keep.append(scenario)

        # If the scenario name doesn't have at least 7 parts, keep it (or adjust the logic as needed)

    # Create a new IamDataFrame with the desired scenarios
    filtered_iamdf = combined_operational_iamdf.filter(scenario=scenarios_to_keep)

    return filtered_iamdf


def drop_inconsistent_embodied_scenarios(combined_embodied_iamdf):
    # Define a dictionary of requirements for different parts of the scenario name
    requirement_dict = {
        'BAU_construction': 'HDS',  # {parts[0]}_{parts[1]} == part[10]
        'S2_construction': 'MDS',  # {parts[0]}_{parts[1]} == part[10]
        'BAU_RE2020_upfront': 'BAU_RE2020_WLC',  # {parts[2]}_{parts[3]}_{parts[4]} == 'BAU_RE2020_upfront'
        'Optimist_RE2020_upfront': 'Optimist_RE2020_WLC'
        # {parts[2]}_{parts[3]}_{parts[4]} == 'Optimist_RE2020_upfront'
        # Add more requirements as needed
    }

    # Create an empty list to store scenarios to keep
    scenarios_to_keep = []

    # Iterate through scenario names
    for scenario in combined_embodied_iamdf.data['scenario']:
        # Split the scenario name into parts using underscores
        parts = scenario.split("_")

        # Check if the scenario name has at least 11 parts (parts[0] to parts[10])
        if len(parts) >= 11:
            # Check if text0_text1 is in the requirement_dict
            if f'{parts[0]}_{parts[1]}' in requirement_dict:
                # Check if text5 matches the requirement for text0_text1
                if parts[10] == requirement_dict[f'{parts[0]}_{parts[1]}']:
                    # Check if the combination of parts[2]_parts[3]_parts[4] matches the requirement
                    matching_requirement = requirement_dict.get(f'{parts[2]}_{parts[3]}_{parts[4]}', None)
                    if matching_requirement is not None and f'{parts[7]}_{parts[8]}_{parts[9]}' == matching_requirement:
                        scenarios_to_keep.append(scenario)

    # Create a new IamDataFrame with the desired scenarios
    filtered_iamdf = combined_embodied_iamdf.filter(scenario=scenarios_to_keep)

    return filtered_iamdf