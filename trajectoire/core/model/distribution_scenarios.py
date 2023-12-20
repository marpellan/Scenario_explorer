# -*- coding: utf-8 -*-

# ========================================
# External imports
# ========================================

import pandas as pd
import numpy as np
from importlib import resources
import matplotlib.pyplot as plt
import pyam

# ========================================
# External CSTB imports
# ========================================


# ========================================
# Internal imports
# ========================================


from trajectoire import results

# ========================================
# Constants
# ========================================

RESULT_DIRECTORY_PATH = resources.files(results)
RESULT_NEW_NRJ_PATH = RESULT_DIRECTORY_PATH / "results_new_nrj"
RESULT_NEW_GES_PATH = RESULT_DIRECTORY_PATH / "results_new_ges"


# === INPUT DATA ===
## Scenarios en nombre de logement
### HRS = 750 000 logements rénovés par an en moyenne sur la période 2020-2050 = 23 250 000 = 67% du parc rénové sur la période
### MRS = 250 000 logements rénovés par an en moyenne sur la période 2020-2050 = 7 750 000 = 22% du parc rénové sur la période

## Scenarios en nombre de MI et LC
### On calcule la part de MI et LC en fonction de leur part actuel dans le parc
#### = 45% de LC et 55% de MI

## Scenarios en surface MI et LC
### On calcule la surface correspondante en multipliant le nombre de logement par la surface moyenne par type de logement
#### = 103m2 pour MI et 58m2 pour LC

## Chiffres 2020 à 2022
### On prend les chiffres BBC uniquement du fait de notre logique de modélisation qui commence par des réno lourdes
### https://www.effinergie.org/web/images/attach/base_doc/3318/20230719_tableau%20de%20bord.pdf
#### 31 000 logements en 2020, 41 000 logements en 2021, 43 000 logements en 2022

# === FUNCTIONS ===

def generate_renovation_distributions(initial_renovations, total_renovations, title=None):
    # Creating the initial data
    years = np.arange(2020, 2051)
    yearly_renovations = np.zeros(len(years))
    yearly_renovations[:3] = initial_renovations  # Given renovations for 2020, 2021, 2022

    # Creating a DataFrame
    df = pd.DataFrame({
        'Years': years,
        'Yearly_renovation': yearly_renovations
    })

    # Calculating Constant Renovation Distribution
    renovations_done = df['Yearly_renovation'].sum()
    renovations_to_distribute = total_renovations - renovations_done
    df.loc[:2, 'Constant'] = df.loc[:2, 'Yearly_renovation']
    df.loc[3:, 'Constant'] = renovations_to_distribute / len(years[3:])
    df['Cum_Constant'] = df['Constant'].cumsum()

    # Creating Linear Increase Distribution
    df.loc[:2, 'Linear'] = df.loc[:2, 'Yearly_renovation']
    linear_increase_cdf = np.linspace(df.loc[2, 'Yearly_renovation'], renovations_to_distribute, len(years[3:]) + 1)
    linear_increase_cdf = np.cumsum(linear_increase_cdf)
    linear_increase_cdf = linear_increase_cdf / linear_increase_cdf[-1] * renovations_to_distribute
    linear_increase_values = np.diff(linear_increase_cdf)
    df.loc[3:, 'Linear'] = np.maximum.accumulate(
        np.maximum(linear_increase_values, df.loc[2, 'Linear']))
    df['Cum_Linear'] = df['Linear'].cumsum()

    # Creating Increase with Plateau Distribution
    df.loc[:2, 'Plateau'] = df.loc[:2, 'Yearly_renovation']
    plateau_cdf = np.linspace(df.loc[2, 'Yearly_renovation'], renovations_to_distribute,
                              len(years[3:]) - len(years[years > 2035]) + 1)
    plateau_cdf = np.concatenate([plateau_cdf, np.full(len(years[years > 2035]), plateau_cdf[-1])])
    plateau_cdf = np.cumsum(plateau_cdf)
    plateau_cdf = plateau_cdf / plateau_cdf[-1] * renovations_to_distribute
    plateau_values = np.diff(plateau_cdf)
    df.loc[3:, 'Plateau'] = np.maximum.accumulate(np.maximum(plateau_values, df.loc[2, 'Plateau']))
    df['Cum_Plateau'] = df['Plateau'].cumsum()

    # Plotting Yearly Values
    plt.figure(figsize=(8, 6))
    plt.plot(df['Years'], df['Constant'], label='Constant', marker='o')
    plt.plot(df['Years'], df['Linear'], label='Linear', marker='o')
    plt.plot(df['Years'], df['Plateau'], label='Plateau', marker='o')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Number of dwellings')
    plt.legend()
    #plt.grid(True)
    plt.show()

    # Plotting Cumulative Values
    plt.figure(figsize=(8, 6))
    plt.plot(df['Years'], df['Cum_Constant'], label='Cumulative Constant', marker='o')
    plt.plot(df['Years'], df['Cum_Linear'], label='Cumulative Linear', marker='o')
    plt.plot(df['Years'], df['Cum_Plateau'], label='Cumulative Plateau', marker='o')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Cumulative number of dwellings')
    plt.legend()
    #plt.grid(True)
    plt.show()

    # Displaying final DataFrame
    return df


def distribute_dwellings(df):
    # Ratios and average surfaces
    collective_ratio = 0.45
    individual_ratio = 0.55
    collective_surface = 58
    individual_surface = 103

    # Creating new columns for collective and individual dwellings
    for distribution in ['Constant', 'Linear', 'Plateau']:
        df[f'Collective {distribution}'] = df[distribution] * collective_ratio
        df[f'Individual {distribution}'] = df[distribution] * individual_ratio

        df[f'Collective {distribution} Surface'] = df[f'Collective {distribution}'] * collective_surface
        df[f'Individual {distribution} Surface'] = df[f'Individual {distribution}'] * individual_surface

    return df

def pyam_distributions(df):
    # Initialize an empty DataFrame to store the transformed data
    transformed_data = pd.DataFrame()

    # Constants
    model = 'CSTB'
    region = 'France'

    # Iterate through each column in the original DataFrame (excluding 'Years' and 'scenario')
    for column in df.columns[2:]:

        # Scenario
        scenario = column.replace('Collective ', '').replace('Individual ', '').strip()
        scenario = scenario.replace(' Surface', '').replace('Surface ', '').strip()

        # Variable and Unit
        if 'Collective' in column:
            variable = 'Collective_dwelling'
        elif 'Individual' in column:
            variable = 'Individual_dwelling'
        else:
            variable = 'Residential_dwelling'

        if 'Surface' in column.strip():
            unit = 'Surface in m2'
        else:
            unit = 'Number of dwelling'

        # Create a DataFrame for the current column
        temp_df = pd.DataFrame({
            'model': model,
            'region': region,
            'scenario': scenario,
            'variable': variable,
            'unit': unit,
            'year': df['Years'],
            'value': df[column]
        })

        # Add the transformed data to the new DataFrame
        transformed_data = pd.concat([transformed_data, temp_df], ignore_index=True)

    # Convert to a Pyam DataFrame
    pyam_data = pyam.IamDataFrame(transformed_data)

    return pyam_data


# === APPLICATIONS ===
## High renovation scenario
df_hrs= generate_renovation_distributions(initial_renovations=[31000,41000,43000],
                                          total_renovations=23250000,
                                          title='High Renovation Scenario')
df_hrs = distribute_dwellings(df_hrs)
df_hrs = pyam_distributions(df_hrs)
df_hrs.to_csv(RESULT_NEW_NRJ_PATH / 'high_renovation_scenario_distribution.csv', sep=',')

## Middle renovation scenario
df_mrs= generate_renovation_distributions(initial_renovations=[31000,41000,43000],
                                          total_renovations=7750000,
                                          title='Middle Renovation Scenario')
df_mrs = distribute_dwellings(df_mrs)
df_mrs = pyam_distributions(df_mrs)
df_mrs.to_csv(RESULT_NEW_NRJ_PATH / 'middle_renovation_scenario_distribution.csv', sep=',')