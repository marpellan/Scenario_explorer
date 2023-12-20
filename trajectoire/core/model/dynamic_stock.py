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
# ========================================
# Variables
# ========================================

# === LOAD STOCK DATA ===
df_stock_2020 = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "dwelling_stock_per_epc" / "DOUBLE" / "dwelling_stock_before_renovation.csv", sep=";", index_col=[0,1])

# === LOAD SCENARIOS ===
### Renovation pace in pyam format
renovation = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "renovation" / "renovation_scenarios.csv", sep=";")
demolition = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "demolition" / "demolition_scenarios.csv", sep=";")


### Pivot table from the targeting scenarios
pt_col_reno_hrs_fossil = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "collective_dwelling" / "fossil" / "pt_surface_col_hrs_fossil.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs_fossil = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "individual_dwelling" / "fossil" / "pt_surface_ind_hrs_fossil.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_col_reno_mrs_fossil = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "collective_dwelling" / "fossil" / "pt_surface_col_hrs_fossil.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs_fossil = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "individual_dwelling" / "fossil" / "pt_surface_ind_hrs_fossil.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

pt_col_reno_hrs_remaining = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "collective_dwelling" / "remaining" / "pt_surface_col_mrs_remaining.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs_remaining = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "individual_dwelling" / "remaining" / "pt_surface_ind_mrs_remaining.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_col_reno_mrs_remaining = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "collective_dwelling" / "remaining" / "pt_surface_col_mrs_remaining.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs_remaining = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "DOUBLE" / "individual_dwelling" / "remaining" / "pt_surface_ind_mrs_remaining.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)


### Pivot table from the High Renovation Scenarios
pt_col_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "collective_dwelling" / "pt_surface_col_hrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_hrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"individual_dwelling" / "pt_surface_ind_hrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the Medium Renovation Scenarios
pt_col_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"collective_dwelling" / "pt_surface_col_mrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_reno_mrs = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"individual_dwelling" / "pt_surface_ind_mrs.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the High Demolition Scenarios
pt_col_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"collective_dwelling" / "pt_surface_col_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_hds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" /"individual_dwelling" / "pt_surface_ind_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)

### Pivot table from the Medium Demolition Scenarios
pt_col_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "collective_dwelling" / "pt_surface_col_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)
pt_ind_dem_mds = pd.read_csv(CALIBRATION_DIRECTORY_PATH / "pivot_tables" / "scenarios" / "NRJ" / "individual_dwelling" / "pt_surface_ind_hds.csv", sep=";", index_col=[0,1], header=[0,1], skipinitialspace=True)


# === LOAD SCENARIOS ===


# ========================================
# Classes
# ========================================


# ========================================
# Constants
# ========================================

etiquette_to_median_ep = {
    'A': 35,
    'B': 90.5,
    'C': 145.5,
    'D': 215.5,
    'E': 290.5,
    'F': 375.5,
    'G': 460.5
}

energy_vector_priority = ['ef_kwh_charbon', 'ef_kwh_fioul', 'ef_kwh_gpl', 'ef_kwh_gaz', 'ef_kwh_elec', 'ef_kwh_bois', 'ef_kwh_rcu']

median_consumptions_dpe_nrj = {
    'Collective_dwelling': {'A': 48, 'B': 82.3, 'C': 116.5, 'D': 177.7, 'E': 248, 'F': 300.3, 'G': 386.1},
    'Individual_dwelling': {'A': 59.8, 'B': 92.9, 'C': 136.2, 'D': 202.4, 'E': 261.6, 'F': 314.1, 'G': 420.8}
}

median_consumptions_dpe_ges = {
    'Collective_dwelling': {'A': 180, 'B': 238, 'C': 268, 'D': 301, 'E': 315, 'F': 338, 'G': 515},
    'Individual_dwelling': {'A': 140, 'B': 188, 'C': 190, 'D': 195, 'E': 207, 'F': 251, 'G': 408}
}


median_consumptions_dpe_double = {
    'Collective_dwelling': {'A': 72, 'B': 104, 'C': 108, 'D': 166, 'E': 228, 'F': 311, 'G': 486},
    'Individual_dwelling': {'A': 76, 'B': 117, 'C': 135, 'D': 203, 'E': 254, 'F': 286, 'G': 477}
}


# ========================================
# Functions DYNAMIC STOCK
# ========================================

def update_pivot_table_reno(pivot_table_reno, target_number, start_row=6):
    if not isinstance(pivot_table_reno, pd.DataFrame) or not isinstance(pivot_table_reno.index, pd.MultiIndex):
        raise ValueError("Input must be a DataFrame with a MultiIndex.")

    # Create a copy of the DataFrame to perform updates
    updated_df = pivot_table_reno.copy()

    total = 0
    changes = []  # List to store changes made
    changes_df = pd.DataFrame(0, index=pivot_table_reno.index, columns=pivot_table_reno.index)
    changes_df = changes_df.astype(float)

    for i in range(start_row, -1, -1):
        for j in range(len(updated_df.columns)):
            index = updated_df.index[i]
            column = updated_df.columns[j]
            value = updated_df.loc[index, column]

            if total + int(value) <= target_number:
                total += value
                if value != 0: ## ADDED
                    changes.append((index, column, value))  # Store change information
                changes_df.loc[index, column] = value  # Update changes_df with value
                updated_df.loc[index, column] = 0
                if total == target_number:
                    return updated_df, changes_df, changes
            else:
                diff = target_number - total
                removed_amount = value - diff
                if diff != 0: # ADDED
                    changes.append((index, column, diff))  # Store change information
                changes_df.loc[index, column] = diff  # Update changes_df with diff
                updated_df.loc[index, column] -= diff
                total = target_number
                return updated_df, changes_df, changes

    return updated_df, changes_df, changes


def update_pivot_table_reno_targeting(pivot_table_reno_fossil_fuels, pivot_table_reno_remaining, target_number, start_row=6):
    # First, try to meet the target by renovating buildings with high fossil fuel consumption
    updated_df_fossil_fuels, changes_df_fossil_fuels, changes_fossil_fuels = \
        update_pivot_table_reno(pivot_table_reno_fossil_fuels, target_number, start_row)

    # Calculate the remaining target after the first round of renovations
    renovated_surface_fossil_fuels = sum([change[2] for change in changes_fossil_fuels])
    remaining_target = max(0, target_number - renovated_surface_fossil_fuels)

    # If there's still a target to meet, renovate the remaining buildings
    if remaining_target > 0:
        updated_df_remaining, changes_df_remaining, changes_remaining = \
            update_pivot_table_reno(pivot_table_reno_remaining, remaining_target, start_row)
    else:
        updated_df_remaining = pivot_table_reno_remaining.copy()
        changes_df_remaining = pd.DataFrame(0, index=pivot_table_reno_remaining.index, columns=pivot_table_reno_remaining.columns)
        changes_remaining = []

    # Combine results from both rounds of renovations
    #updated_df_combined = updated_df_fossil_fuels + updated_df_remaining
    changes_df_combined = changes_df_fossil_fuels + changes_df_remaining
    changes_combined = changes_fossil_fuels + changes_remaining

    return updated_df_fossil_fuels, updated_df_remaining, changes_df_combined, changes_combined


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
                if value != 0: ## ADDED
                    changes.append((index, column, value))  # Store change information
                changes_df.loc[index, column] = value  # Update changes_df with value
                updated_df.loc[index, column] = 0
                if total == target_number:
                    return updated_df, changes_df, changes
            else:
                diff = target_number - total
                removed_amount = value - diff
                if diff != 0: # ADDED
                    changes.append((index, column, diff))  # Store change information
                changes_df.loc[index, column] = diff  # Update changes_df with diff
                updated_df.loc[index, column] -= diff
                total = target_number
                return updated_df, changes_df, changes

    return updated_df, changes_df, changes


def update_dwelling_stock_nrj(stock_df, changes_renovation, changes_demolition):
    updated_dwelling_stock_df = stock_df.copy()

    # MAJ des surfaces
    for change in changes_renovation + changes_demolition:
        from_index = change[0]
        to_index = change[1] if change in changes_renovation else None
        amount = change[2]

        #print(f"Before update: {from_index} has {updated_dwelling_stock_df.loc[from_index, 'sum_shab']}m² available.")
        #print(f"Attempting to subtract: {amount}m² for {'renovation' if change in changes_renovation else 'demolition'}.")

        if updated_dwelling_stock_df.loc[from_index, 'sum_shab'] >= amount:
            updated_dwelling_stock_df.loc[from_index, 'sum_shab'] -= amount
            if to_index:
                updated_dwelling_stock_df.loc[to_index, 'sum_shab'] += amount
                # Print the new surface after update
            #print(f"After update: {from_index} now has {updated_dwelling_stock_df.loc[from_index, 'sum_shab']}m².")
        else:
            #print(
                #f"Attention : Pas assez de surface pour effectuer la rénovation/démolition pour {from_index} to {to_index}. Ajustement à la surface disponible.")
            amount = updated_dwelling_stock_df.loc[from_index, 'sum_shab']
            updated_dwelling_stock_df.loc[from_index, 'sum_shab'] = 0
            if to_index:
                updated_dwelling_stock_df.loc[to_index, 'sum_shab'] += amount

    # Définition des valeurs médianes de consommation en fonction du type de logement
    median_consumptions = median_consumptions_dpe_double

    energy_vectors_order = ['ep_kwh_charbon', 'ep_kwh_fioul', 'ep_kwh_gpl', 'ep_kwh_gaz', 'ep_kwh_elec', 'ep_kwh_bois',
                            'ep_kwh_rcu']
    ef_energy_vectors_order = ['ef_kwh_charbon', 'ef_kwh_fioul', 'ef_kwh_gpl', 'ef_kwh_gaz', 'ef_kwh_elec',
                               'ef_kwh_bois', 'ef_kwh_rcu']

    for change in changes_renovation + changes_demolition:
        from_index = change[0]
        dwelling_type = from_index[0]
        label = from_index[1]
        amount = change[2] * median_consumptions[dwelling_type][label]

        # Mise à jour des consommations d'énergie primaire
        for energy_vector in energy_vectors_order:
            current_energy_amount = updated_dwelling_stock_df.loc[from_index, energy_vector]
            if amount > current_energy_amount:
                updated_dwelling_stock_df.loc[from_index, energy_vector] = 0
                amount -= current_energy_amount
            else:
                updated_dwelling_stock_df.loc[from_index, energy_vector] -= amount
                amount = 0
                break

        # Pour les rénovations, ajouter aux étiquettes après rénovation
        if change in changes_renovation:
            to_index = change[1]
            dwelling_type = to_index[0]
            label = to_index[1]
            amount = change[2] * median_consumptions[dwelling_type][label]
            if 'Collective_dwelling' in to_index[0]:
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_elec'] += 0.8 * amount
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_rcu'] += 0.2 * amount
            elif 'Individual_dwelling' in to_index[0]:
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_elec'] += 0.8 * amount
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_bois'] += 0.2 * amount

    # Mise à jour de ep_kwh_calibrated
    updated_dwelling_stock_df['ep_kwh_calibrated'] = updated_dwelling_stock_df[energy_vectors_order].sum(axis=1)
    updated_dwelling_stock_df['ep_kwh_calibrated'] = updated_dwelling_stock_df['ep_kwh_calibrated'].apply(
        lambda x: 0 if abs(x) < 1e-4 else x)

    # Calcul du delta
    df_delta = updated_dwelling_stock_df - stock_df

    # Mise à jour des consommations d'énergie finale (EF)
    for vector in energy_vectors_order:
        if 'elec' in vector:
            df_delta['ef_kwh_elec'] = df_delta['ep_kwh_elec'] / 2.3
        else:
            df_delta['ef_kwh_' + vector[7:]] = df_delta['ep_kwh_' + vector[7:]]

    # Mise à jour de ef_kwh_calibrated
    df_delta['ef_kwh_calibrated'] = df_delta[ef_energy_vectors_order].sum(axis=1)

    # Calcul du nouveau dataframe
    df_updated = stock_df + df_delta

    return df_updated, df_delta


def run_dynamic_simulation_double(pivot_table_reno_fossil, pivot_table_remaining, pivot_table_dem, surface_df,
                           renovation_scenario=None, renovation_variable=None,
                           demolition_scenario=None, demolition_variable=None):
    updated_reno_dfs_fossil = {}  # To store updated pivot tables for renovation for each year
    updated_reno_dfs_remaining = {}
    updated_dem_dfs = {}  # To store updated pivot tables for demolition for each year
    updated_surface_dfs = {}  # To store updated stock dataFrames for each year
    delta_dfs = {}  # To store the difference between the updated and original dataFrames

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

        updated_reno_df_fossil, updated_reno_df_remaining, changes_reno_df, changes_reno = update_pivot_table_reno_targeting(pivot_table_reno_fossil, pivot_table_remaining, target_number_reno)
        updated_dem_df, changes_dem_df, changes_dem = update_pivot_table_dem(pivot_table_dem, target_number_dem)
        updated_surface_df, df_delta = update_dwelling_stock_nrj(surface_df, changes_reno, changes_dem)

        updated_reno_dfs_fossil[year + 1] = updated_reno_dfs_fossil
        updated_reno_dfs_remaining[year + 1] = updated_reno_dfs_remaining
        updated_dem_dfs[year + 1] = updated_dem_df
        updated_surface_dfs[year + 1] = updated_surface_df
        delta_dfs[year] = df_delta

        changes_reno_lists[year] = changes_reno
        changes_dem_lists[year] = changes_dem

        changes_reno_dfs[year] = changes_reno_df
        changes_dem_dfs[year] = changes_dem_df

        # Use updated data for the next year's calculations
        pivot_table_reno_fossil = updated_reno_df_fossil
        pivot_table_remaining = updated_reno_df_remaining
        pivot_table_dem = updated_dem_df
        surface_df = updated_surface_df

    return (updated_surface_dfs, changes_reno_lists, changes_dem_lists, delta_dfs)


def run_dynamic_simulation_nrj(pivot_table_reno, pivot_table_dem, surface_df,
                           renovation_scenario=None, renovation_variable=None,
                           demolition_scenario=None, demolition_variable=None):
    updated_reno_dfs = {}  # To store updated pivot tables for renovation for each year
    updated_dem_dfs = {}  # To store updated pivot tables for demolition for each year
    updated_surface_dfs = {}  # To store updated stock dataFrames for each year
    delta_dfs = {}  # To store the difference between the updated and original dataFrames

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
        updated_surface_df, df_delta = update_dwelling_stock_nrj(surface_df, changes_reno, changes_dem)

        updated_reno_dfs[year + 1] = updated_reno_df
        updated_dem_dfs[year + 1] = updated_dem_df
        updated_surface_dfs[year + 1] = updated_surface_df
        delta_dfs[year] = df_delta

        changes_reno_lists[year] = changes_reno
        changes_dem_lists[year] = changes_dem

        changes_reno_dfs[year] = changes_reno_df
        changes_dem_dfs[year] = changes_dem_df

        # Use updated data for the next year's calculations
        pivot_table_reno = updated_reno_df
        pivot_table_dem = updated_dem_df
        surface_df = updated_surface_df

    return (updated_surface_dfs, changes_reno_lists, changes_dem_lists, delta_dfs)



def update_dwelling_stock_ges(stock_df, changes_renovation, changes_demolition):
    updated_dwelling_stock_df = stock_df.copy()

    # MAJ des surfaces
    for change in changes_renovation + changes_demolition:
        from_index = change[0]
        to_index = change[1] if change in changes_renovation else None
        amount = change[2]

        print(f"Before update: {from_index} has {updated_dwelling_stock_df.loc[from_index, 'sum_shab']}m² available.")
        print(f"Attempting to subtract: {amount}m² for {'renovation' if change in changes_renovation else 'demolition'}.")

        if updated_dwelling_stock_df.loc[from_index, 'sum_shab'] >= amount:
            updated_dwelling_stock_df.loc[from_index, 'sum_shab'] -= amount
            if to_index:
                updated_dwelling_stock_df.loc[to_index, 'sum_shab'] += amount
                # Print the new surface after update
            print(f"After update: {from_index} now has {updated_dwelling_stock_df.loc[from_index, 'sum_shab']}m².")
        else:
            print(
                f"Attention : Pas assez de surface pour effectuer la rénovation/démolition pour {from_index} to {to_index}. Ajustement à la surface disponible.")
            amount = updated_dwelling_stock_df.loc[from_index, 'sum_shab']
            updated_dwelling_stock_df.loc[from_index, 'sum_shab'] = 0
            if to_index:
                updated_dwelling_stock_df.loc[to_index, 'sum_shab'] += amount

    # Définition des valeurs médianes de consommation en fonction du type de logement
    median_consumptions = median_consumptions_dpe_ges

    energy_vectors_order = ['ep_kwh_charbon', 'ep_kwh_fioul', 'ep_kwh_gpl', 'ep_kwh_gaz', 'ep_kwh_elec', 'ep_kwh_bois',
                            'ep_kwh_rcu']
    ef_energy_vectors_order = ['ef_kwh_charbon', 'ef_kwh_fioul', 'ef_kwh_gpl', 'ef_kwh_gaz', 'ef_kwh_elec',
                               'ef_kwh_bois', 'ef_kwh_rcu']

    for change in changes_renovation + changes_demolition:
        from_index = change[0]
        dwelling_type = from_index[0]
        label = from_index[1]
        amount = change[2] * median_consumptions[dwelling_type][label]

        # Mise à jour des consommations d'énergie primaire
        for energy_vector in energy_vectors_order:
            current_energy_amount = updated_dwelling_stock_df.loc[from_index, energy_vector]
            if amount > current_energy_amount:
                updated_dwelling_stock_df.loc[from_index, energy_vector] = 0
                amount -= current_energy_amount
            else:
                updated_dwelling_stock_df.loc[from_index, energy_vector] -= amount
                amount = 0
                break

        # Pour les rénovations, ajouter aux étiquettes après rénovation
        if change in changes_renovation:
            to_index = change[1]
            dwelling_type = to_index[0]
            label = to_index[1]
            amount = change[2] * median_consumptions[dwelling_type][label]
            if 'Collective_dwelling' in to_index[0]:
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_elec'] += 0.8 * amount
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_rcu'] += 0.2 * amount
            elif 'Individual_dwelling' in to_index[0]:
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_elec'] += 0.8 * amount
                updated_dwelling_stock_df.loc[to_index, 'ep_kwh_bois'] += 0.2 * amount

    # Mise à jour de ep_kwh_calibrated
    updated_dwelling_stock_df['ep_kwh_calibrated'] = updated_dwelling_stock_df[energy_vectors_order].sum(axis=1)
    updated_dwelling_stock_df['ep_kwh_calibrated'] = updated_dwelling_stock_df['ep_kwh_calibrated'].apply(
        lambda x: 0 if abs(x) < 1e-4 else x)

    # Calcul du delta
    df_delta = updated_dwelling_stock_df - stock_df

    # Mise à jour des consommations d'énergie finale (EF)
    for vector in energy_vectors_order:
        if 'elec' in vector:
            df_delta['ef_kwh_elec'] = df_delta['ep_kwh_elec'] / 2.3
        else:
            df_delta['ef_kwh_' + vector[7:]] = df_delta['ep_kwh_' + vector[7:]]

    # Mise à jour de ef_kwh_calibrated
    df_delta['ef_kwh_calibrated'] = df_delta[ef_energy_vectors_order].sum(axis=1)

    # Calcul du nouveau dataframe
    df_updated = stock_df + df_delta

    return df_updated, df_delta


def run_dynamic_simulation_ges(pivot_table_reno, pivot_table_dem, surface_df,
                           renovation_scenario=None, renovation_variable=None,
                           demolition_scenario=None, demolition_variable=None):
    updated_reno_dfs = {}  # To store updated pivot tables for renovation for each year
    updated_dem_dfs = {}  # To store updated pivot tables for demolition for each year
    updated_surface_dfs = {}  # To store updated stock dataFrames for each year
    delta_dfs = {}  # To store the difference between the updated and original dataFrames

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
        updated_surface_df, df_delta = update_dwelling_stock_ges(surface_df, changes_reno, changes_dem)

        updated_reno_dfs[year + 1] = updated_reno_df
        updated_dem_dfs[year + 1] = updated_dem_df
        updated_surface_dfs[year + 1] = updated_surface_df
        delta_dfs[year] = df_delta

        changes_reno_lists[year] = changes_reno
        changes_dem_lists[year] = changes_dem

        changes_reno_dfs[year] = changes_reno_df
        changes_dem_dfs[year] = changes_dem_df

        # Use updated data for the next year's calculations
        pivot_table_reno = updated_reno_df
        pivot_table_dem = updated_dem_df
        surface_df = updated_surface_df

    return (updated_surface_dfs, changes_reno_lists, changes_dem_lists, delta_dfs)


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


def create_iamdf_stock(updated_surface_dfs, scenario_name, variable_name):
    iamdf_list = []

    for year, df in updated_surface_dfs.items():
        # Step 1: Melt the DataFrame
        df_melted = pd.melt(df.reset_index(),
                            id_vars=['ffo_bat_usage_niveau_1_txt', 'simu_dpe_etiquette_dpe_initial_map', 'sum_shab'],
                            var_name='original_variable',
                            value_name='value')

        # Filter to include only 'ef_kwh_...' variables
        df_melted = df_melted[df_melted['original_variable'].str.startswith('ef_kwh_')]

        # Step 2: Set other required columns
        df_melted['model'] = 'CSTB'
        df_melted['scenario'] = scenario_name
        df_melted['region'] = 'France'
        df_melted['unit'] = 'kWh_ef'
        df_melted['year'] = year
        df_melted['variable'] = variable_name  # Set the variable name

        # Setting 'ef_nrj' and 'dpe'
        df_melted['ef_nrj'] = df_melted['original_variable'].str.replace('ef_kwh_', '')
        df_melted['dpe'] = df_melted['simu_dpe_etiquette_dpe_initial_map']

        # Selecting required columns for IamDataFrame
        df_iam = df_melted[['model', 'scenario', 'region', 'variable', 'unit', 'value', 'ef_nrj', 'dpe', 'year']]

        # Adding to the list
        iamdf_list.append(pyam.IamDataFrame(df_iam))

    # Step 3: Concatenate the results into a single IamDataFrame
    iamdf = pyam.concat(iamdf_list)

    return iamdf


def add_residential_dwelling_stock(iamdf):
    # Make a copy of the dataframe
    iamdf_new = iamdf.copy().data

    # Filter rows for 'Collective_dwelling' and 'Individual_dwelling'
    collective_individual_df = iamdf_new[iamdf_new['variable'].isin(['Collective_dwelling', 'Individual_dwelling'])]

    # Group by 'scenario', 'year', 'model', 'region', and 'unit' and calculate the residential_dwelling_value for each group
    group_sum = collective_individual_df.groupby(['scenario', 'year', 'model', 'region', 'unit', 'dpe', 'ef_nrj'])['value'].sum()

    # Convert the group_sum Series to a DataFrame
    group_sum_df = group_sum.reset_index()

    # Add 'Residential_dwelling' as the variable for each group
    group_sum_df['variable'] = 'Residential_dwelling'

    # Append the group_sum_df to the original dataframe
    iamdf_new = pd.concat(
        [iamdf_new, group_sum_df[['scenario', 'year', 'model', 'region', 'unit', 'variable', 'dpe', 'ef_nrj', 'value']]],
        ignore_index=True)

    iamdf_new_output = pyam.IamDataFrame(iamdf_new)

    return iamdf_new_output

# ========================================
# Functions DYNAMIC STOCK * GHGE
# ========================================

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


def calculate_combined_operational_stock_iamdf(stock, ghge):
    '''
    Calculates operational GHGE of the stock,
    by multiplying kWh of energy carriers by the corresponding GHGE factors.

    Returns a unique pyam dataframe
    '''

    result_list = []

    for stock_scenario in stock.scenario:
        for stock_variable in stock.variable:
            for stock_dpe in stock.dpe:
                for ef_nrj in stock.ef_nrj:  # ['bois', 'charbon', 'elec', 'fioul', 'gaz', 'gpl', 'rcu']
                    for ghge_scenario in ghge.scenario:

                        # Make sure we're matching the energy carrier in both dataframes
                        stock_df = stock.filter(scenario=stock_scenario, variable=stock_variable, dpe=stock_dpe, ef_nrj=ef_nrj).data
                        ghge_df = ghge.filter(scenario=ghge_scenario, variable=ef_nrj).data

                        # Check if there is data to multiply, otherwise skip to the next iteration
                        if stock_df.empty or ghge_df.empty:
                            continue

                        # Calculate GHGE in MtCO2eq
                        value_df = stock_df['value'] * ghge_df['value'] / 10e8

                        # Construct the result dataframe
                        result_df = stock_df.copy()
                        result_df['value'] = value_df
                        result_df['scenario'] = f'{stock_scenario}_{ghge_scenario}'
                        result_df['variable'] = stock_variable
                        result_df['unit'] = 'MtCO2eq'

                        result_list.append(result_df)

    # Convert the results to an IamDataFrame
    results_df = pd.concat(result_list)
    results_iamdf = pyam.IamDataFrame(results_df)

    return results_iamdf


def calculate_combined_ghge_new_construction(construction, embodied_ghge):
    construction_col_list = []
    construction_ind_list = []
    construction_res_list = []

    for construction_scenario in construction.scenario:
        construction_col_df = construction.filter(scenario=construction_scenario, variable='Collective_dwelling', unit='Surface in m2').data
        construction_ind_df = construction.filter(scenario=construction_scenario, variable='Individual_dwelling', unit='Surface in m2').data

        for scenario in embodied_ghge.scenario:
            embodied_ghge_col_df = embodied_ghge.filter(scenario=scenario, variable='Collective_dwelling').data
            embodied_ghge_ind_df = embodied_ghge.filter(scenario=scenario, variable='Individual_dwelling').data

            # Multiply m2 and kgCO2eq/m2
            construction_embodied_col = construction_col_df['value'] * embodied_ghge_col_df['value']/10e8
            construction_embodied_ind = construction_ind_df['value'] * embodied_ghge_ind_df['value']/10e8
            construction_embodied_res = construction_embodied_col + construction_embodied_ind

            scenario_name = f'{construction_scenario}_{scenario}'

            # Create df
            result_col_df = pd.DataFrame(
                {'year': construction_col_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': construction_col_df.variable[0],
                 'unit': 'MtCO2eq',
                 'value': construction_embodied_col})

            result_ind_df = pd.DataFrame(
                {'year': construction_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': construction_ind_df.variable[0],
                 'unit': 'MtCO2eq',
                 'value': construction_embodied_ind})

            result_res_df = pd.DataFrame(
                {'year': construction_ind_df['year'],
                 'model': 'CSTB',
                 'scenario': scenario_name,
                 'region': 'France',
                 'variable': 'Residential_dwelling',
                 'unit': 'MtCO2eq',
                 'value': construction_embodied_res})

            construction_col_list.append(result_col_df)
            construction_ind_list.append(result_ind_df)
            construction_res_list.append(result_res_df)

    combined_construction_embodied_col_df = pd.concat(construction_col_list)
    combined_construction_embodied_ind_df = pd.concat(construction_ind_list)
    combined_construction_embodied_res_df = pd.concat(construction_res_list)

    # Concatenate both DataFrames into a single IamDataFrame
    combined_construction_embodied_df = pd.concat([combined_construction_embodied_col_df, combined_construction_embodied_ind_df, combined_construction_embodied_res_df])
    combined_construction_embodied_iamdf = pyam.IamDataFrame(combined_construction_embodied_df)

    return combined_construction_embodied_iamdf


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

def operational_all_stock(construction, remaining,
                          construction_scenario=None, remaining_scenario=None,
                          created_scenario_name=None):


    df_construction = construction.filter(scenario=construction_scenario).data
    df_remaining = remaining.filter(scenario=remaining_scenario).data

    # We drop the sum in remaining
    #df_remaining.drop(df_remaining.loc[df_remaining['dpe'] == 'sum'].index, inplace=True)

    # We add blabala
    df_construction['year'] = df_construction['year'] + 1
    #df_construction['dpe'] = 'new'
    #df_construction['ef_nrj'] = 'new_mix'

    # Concatenate the 2 dataframes together, since they have the same variables and scenario names
    combined_programmation_df = pd.concat([df_construction, df_remaining])

    # We put the same scenario name
    combined_programmation_df['scenario'] = created_scenario_name

    # Create a single IamDataFramer containing all the data
    combined_programmation_iamdf = pyam.IamDataFrame(combined_programmation_df)

    return combined_programmation_iamdf


def operational_all_stock_combined(construction, remaining):
    combined_list = []

    for construction_scenario in construction.scenario:
        df_construction = construction.filter(scenario=construction_scenario).data

        # We add blabala
        df_construction['year'] = df_construction['year'] + 1
        #df_construction['dpe'] = 'new'
        #df_construction['ef_nrj'] = 'new_mix'

        for remaining_scenario in remaining.scenario:
            df_remaining = remaining.filter(scenario=remaining_scenario).data

            # We drop the sum in remaining
            #df_remaining.drop(df_remaining.loc[df_remaining['dpe'] == 'sum'].index, inplace=True)

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
        'AMB_construction': 'MDS',
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


def drop_inconsistent_nrj_scenarios(combined_operational_iamdf):
    # Define a dictionary of requirements for text0_text1 and text5
    requirement_dict = {
        'BAU_construction': 'HDS',
        'AMB_construction': 'MDS',
        # Add more requirements as needed
    }

    # Create an empty list to store scenarios to keep
    scenarios_to_keep = []

    # Iterate through scenario names
    for scenario in combined_operational_iamdf.data['scenario']:
        # Split the scenario name into parts using underscores
        parts = scenario.split("_")

        # Check if the scenario name has at least 7 parts (text0_text1_text2_text3_text4_text5_text6)
        if len(parts) >= 5:
            # Check if text0_text1 is in the requirement_dict
            if f'{parts[0]}_{parts[1]}' in requirement_dict:
                # Check if text5 matches the requirement for text0_text1
                if parts[4] == requirement_dict[f'{parts[0]}_{parts[1]}']:
                    scenarios_to_keep.append(scenario)

        # If the scenario name doesn't have at least 7 parts, keep it (or adjust the logic as needed)

    # Create a new IamDataFrame with the desired scenarios
    filtered_iamdf = combined_operational_iamdf.filter(scenario=scenarios_to_keep)

    return filtered_iamdf


def drop_inconsistent_embodied_scenarios(combined_embodied_iamdf):
    requirement_dict = {
        'BAU_construction': [(6, 'HDS'), (3, 'MRS')],
        'AMB_construction': [(6, 'MDS'), (3, 'HRS')]
        # Other specific scenario requirements can be added here if necessary
    }

    scenarios_to_keep = []

    for scenario in combined_embodied_iamdf.data['scenario']:
        parts = scenario.split("_")
        scenario_base = f'{parts[0]}_{parts[1]}'

        keep_scenario = True  # Start by assuming the scenario is valid

        # First, handle the specific requirements for 'BAU_construction' and 'AMB_construction'
        if scenario_base in requirement_dict:
            requirements = requirement_dict[scenario_base]
            for part_index, required_value in requirements:
                if part_index < len(parts) and parts[part_index] != required_value:
                    keep_scenario = False
                    break

        # Then, check if positions 2, 5, and 7 are equal for all scenarios
        if len(parts) > 7 and not (parts[2] == parts[5] == parts[7]):
            keep_scenario = False

        # If the scenario is still considered valid, add it to the list to keep
        if keep_scenario:
            scenarios_to_keep.append(scenario)

    # Filter the dataframe to only include the scenarios we've decided to keep
    filtered_iamdf = combined_embodied_iamdf.filter(scenario=scenarios_to_keep)
    return filtered_iamdf


def drop_inconsistent_embodied_scenarios_2(combined_embodied_iamdf):
    requirement_dict = {
        'BAU_construction': [(6, 'HDS'), (3, 'MRS')], # it is working
        'AMB_construction': [(6, 'MDS'), (3, 'HRS')], # it is working
        'RE2020': [([7, 8, 9], 'RE2020')], # position 2,5,7 = RE2020
        'RE2020++': [([7, 8, 9], 'RE2020++')] # position 2,5,7 = RE2020++
    }

    scenarios_to_keep = []

    for scenario in combined_embodied_iamdf.data['scenario']:
        parts = scenario.split("_")
        scenario_base = f'{parts[0]}_{parts[1]}'

        if scenario_base in requirement_dict:
            requirements = requirement_dict[scenario_base]
            keep_scenario = True

            for part_indices, required_value in requirements:
                if isinstance(part_indices, list):
                    # Concatenate the parts of the scenario name for comparison
                    combined_parts = '_'.join(parts[i] for i in part_indices if i < len(parts))
                    if combined_parts != required_value:
                        keep_scenario = False
                        break
                elif part_indices < len(parts) and parts[part_indices] != required_value:
                    keep_scenario = False
                    break

            if keep_scenario:
                scenarios_to_keep.append(scenario)

    filtered_iamdf = combined_embodied_iamdf.filter(scenario=scenarios_to_keep)
    return filtered_iamdf
