# -*- coding: utf-8 -*-

# ========================================
# External imports
# ========================================

import pandas as pd
import pyam

from importlib import resources

# ========================================
# External CSTB imports
# ========================================


# ========================================
# Internal imports
# ========================================

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

# TODO: A mettre dans constants, si jamais modifié (et en majuscules)
# === LOAD SCENARIOS ===
population = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "new_construction" / "population_scenarios.csv", sep=';')
FApC = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "new_construction" / "m2_per_capita_scenarios.csv", sep=';')
dem = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "demolition" / "demolition_scenarios_bdnb.csv", sep=';')
new = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "new_construction" / "new_surface_scenarios.csv", sep=';')
pct_ind_col_surface = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "new_construction" / "pct_ind_col_new_scenarios.csv", sep=';')
embodied_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_new_construction_scenarios.csv", sep=';')
operational_ghge = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "operational_ghge_new_construction_scenarios.csv", sep=';')

# ========================================
# Classes
# ========================================


# ========================================
# Functions
# ========================================

# === FUNCTIONS TO GET HOUSING NEEDS AND CONSTRUCTION NEEDS FROM POPULATION AND NEEDS GROWTH ===
def calculate_housing_needs(population, FApC,
                                population_scenarios=None, FApC_scenarios=None,
                                FApC_variable=None,
                                unit_name=None):
    '''
    Calculate the housing needs based on the population and the m2/capita scenarios as variables
    Returns an IamDataFrame with the housing needs in m2
    '''

    housing_needs_list = []  # List to store the calculated housing needs for each combination

    for pop_scenario in population_scenarios:
        population_filtered = population.filter(scenario=pop_scenario)
        population_df = population_filtered.as_pandas()

        for FApC_scenario in FApC_scenarios:
            FApC_filtered = FApC.filter(scenario=FApC_scenario, variable=FApC_variable)
            FApC_df = FApC_filtered.as_pandas()

            # Multiply population and FApC timeseries to get housing needs
            housing_needs = population_df['value'] * FApC_df['value']

            # Create a new dataframe with year as a column
            result_df = pd.DataFrame({'year': population_df['year'],
                                      'value': housing_needs})

            # Get the model and scenario names from population and FApC scenarios
            model_name = population_filtered['model'].iloc[0]
            scenario_name = population_filtered['scenario'].iloc[0] + '_' + FApC_filtered['scenario'].iloc[0]

            # Create a new IamDataFrame object from the housing needs timeseries
            housing_needs_iamdf = pyam.IamDataFrame(result_df,
                                                    model=model_name,
                                                    scenario=scenario_name,
                                                    region='France',
                                                    variable='Housing needs',
                                                    unit=unit_name)

            housing_needs_list.append(housing_needs_iamdf)

    # Concatenate all the housing needs IamDataframes into a single IamDataFrame
    combined_housing_needs_iamdf = pyam.concat(housing_needs_list)

    return combined_housing_needs_iamdf


def calculate_construction_needs_from_population_growth(housing_needs_combined_iamdf):
    '''
    Calculate the growth of housing needs based on the rise in demand from population growth and/or m2/capita
    '''

    construction_needs_list = []  # List to store the calculated housing needs for each combination

    for scenarios in housing_needs_combined_iamdf.scenario:
        housing_needs_filtered = housing_needs_combined_iamdf.filter(scenario=scenarios)
        housing_needs_df = housing_needs_filtered.as_pandas()

        # calculate the difference in housing needs from t to t-1
        construction_needs = housing_needs_df['value'].diff().fillna(0)

        # replace negative values with 0
        construction_needs = construction_needs.clip(lower=0)

        # create a new dataframe with year as a column
        result_df = pd.DataFrame({'year': housing_needs_combined_iamdf['year']-1,
                                  'value': construction_needs})

        # drop the first year of result_df e.g. 2019
        result_df = result_df.drop(result_df.index[0])

        # Get the model and scenario names from housing_needs
        model_name = housing_needs_filtered['model'].iloc[0]
        scenario_name = housing_needs_filtered['scenario'].iloc[0]

        # Create a new IamDataFrame object from the housing needs timeseries
        construction_needs_iamdf = pyam.IamDataFrame(result_df,
                                                     model=model_name,
                                                     scenario=scenario_name,
                                                     region='France',
                                                     variable='Construction needs from population growth',
                                                     unit='m2')

        construction_needs_list.append(construction_needs_iamdf)

        # Concatenate all the housing needs IamDataframes into a single IamDataFrame
        combined_construction_needs_iamdf = pyam.concat(construction_needs_list)

    return combined_construction_needs_iamdf


def run_construction_pyam(combined_construction_needs_from_population_growth_iamdf,
                          demolition,
                   dem_scenario=None, construction_needs_scenario=None,
                   pct_ind_col_surface_scenario=None, new_scenario=None,
                     created_scenario_construction_name=None):

    # Create empty list to store results
    effective_construction_list = []
    effective_construction_col_list = []
    effective_construction_ind_list = []
    new_dwelling_list = []
    new_dwelling_col_list = []
    new_dwelling_ind_list = []

    # Iterate over years
    for year in range(2020, 2051):
        # Take demolition and construction needs surface
        demolition_surface = demolition.filter(year=year, scenario=dem_scenario, variable='Residential_dwelling', unit='Surface in m2').data['value'].values[0]
        construction_needs_surface = combined_construction_needs_from_population_growth_iamdf.filter(year=year, variable='Construction needs from population growth', scenario=construction_needs_scenario).data['value'].values[0]

        # Calculate construction needs
        # construction_needs = demolition_surface + construction_needs_surface

        # Calculate construction needs with vacant surface becoming available
        # effective_construction = construction_needs - (stock * vacant_surface.filter(year=year, scenario=vacant_surface_scenario, variable='perc').data['value'].values[0]

        effective_construction = demolition_surface + construction_needs_surface

        # Calculate individual and collective construction surface based on the effective construction
        effective_construction_ind = effective_construction * pct_ind_col_surface.filter(year=year, scenario=pct_ind_col_surface_scenario, variable='Individual_dwelling').data['value'].values[0]
        effective_construction_col = effective_construction * pct_ind_col_surface.filter(year=year, scenario=pct_ind_col_surface_scenario, variable='Collective_dwelling').data['value'].values[0]

        # Convert individual and collective construction surface to number of dwellings
        new_dwelling_ind = effective_construction_ind / new.filter(year=year, scenario=new_scenario, variable='Individual_dwelling').data['value'].values[0]
        new_dwelling_col = effective_construction_col / new.filter(year=year, scenario=new_scenario, variable='Collective_dwelling').data['value'].values[0]
        new_dwelling = new_dwelling_ind + new_dwelling_col

        # Create new dataframes with year as a column

        effective_construction_df = pd.DataFrame({'year': [year],
                                  'value': effective_construction})
        effective_construction_col_df = pd.DataFrame({'year': [year],
                                  'value': effective_construction_col})
        effective_construction_ind_df = pd.DataFrame({'year': [year],
                                  'value': effective_construction_ind})
        new_dwelling_df = pd.DataFrame({'year': [year],
                                  'value': new_dwelling})
        new_dwelling_col_df = pd.DataFrame({'year': [year],
                                  'value': new_dwelling_col})
        new_dwelling_ind_df = pd.DataFrame({'year': [year],
                                  'value': new_dwelling_ind})



        # Create new IamDataFrame object from the housing needs timeseries
        effective_construction_iamdf = pyam.IamDataFrame(effective_construction_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Residential_dwelling',
                                                     unit='Surface in m2')

        effective_construction_col_iamdf = pyam.IamDataFrame(effective_construction_col_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Collective_dwelling',
                                                     unit='Surface in m2')

        effective_construction_ind_iamdf = pyam.IamDataFrame(effective_construction_ind_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Individual_dwelling',
                                                     unit='Surface in m2')

        new_dwelling_iamdf = pyam.IamDataFrame(new_dwelling_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Residential_dwelling',
                                                     unit='Number of dwelling')
        new_dwelling_col_iamdf = pyam.IamDataFrame(new_dwelling_col_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Collective_dwelling',
                                                     unit='Number of dwelling')
        new_dwelling_ind_iamdf = pyam.IamDataFrame(new_dwelling_ind_df,
                                                     model='CSTB',
                                                     scenario=created_scenario_construction_name,
                                                     region='France',
                                                     variable='Individual_dwelling',
                                                     unit='Number of dwelling')

        effective_construction_list.append(effective_construction_iamdf)
        effective_construction_col_list.append(effective_construction_col_iamdf)
        effective_construction_ind_list.append(effective_construction_ind_iamdf)
        new_dwelling_list.append(new_dwelling_iamdf)
        new_dwelling_col_list.append(new_dwelling_col_iamdf)
        new_dwelling_ind_list.append(new_dwelling_ind_iamdf)

        # Concatenate all IamDataframes into a single IamDataFrame
        combined_construction_iamdf = pyam.concat(effective_construction_list +
                                                  effective_construction_col_list +
                                                  effective_construction_ind_list +
                                                  new_dwelling_list +
                                                  new_dwelling_col_list +
                                                  new_dwelling_ind_list
                                                  )

    return combined_construction_iamdf


def run_construction_pyam_2(combined_construction_needs_from_population_growth_iamdf, dem_scenario=None,
                          construction_needs_scenario=None, pct_ind_col_surface_scenario=None, new_scenario=None,
                          created_scenario_construction_name=None):
    # Prepare data containers
    data = {'effective_construction': [], 'effective_construction_col': [], 'effective_construction_ind': [],
            'new_dwelling': [], 'new_dwelling_col': [], 'new_dwelling_ind': []}

    # Pre-filter data outside the loop for efficiency
    dem_pre_filtered = dem.filter(scenario=dem_scenario, variable='Residential_dwelling', unit='Surface in m2')
    construction_needs_pre_filtered = combined_construction_needs_from_population_growth_iamdf.filter(
        scenario=construction_needs_scenario)
    pct_ind_col_surface_pre_filtered = pct_ind_col_surface.filter(scenario=pct_ind_col_surface_scenario)

    # Iterate over years
    for year in range(2020, 2051):
        # Take demolition and construction needs surface
        demolition_surface = dem_pre_filtered.filter(year=year).data['value'].values[0]
        construction_needs_surface = construction_needs_pre_filtered.filter(year=year).data['value'].values[0]

        # Calculate construction needs
        effective_construction = demolition_surface + construction_needs_surface

        # Calculate individual and collective construction surface
        effective_construction_ind = effective_construction * pct_ind_col_surface_pre_filtered.filter(year=year,
                                                                                                      variable='Individual_dwelling').data[
            'value'].values[0]
        effective_construction_col = effective_construction * pct_ind_col_surface_pre_filtered.filter(year=year,
                                                                                                      variable='Collective_dwelling').data[
            'value'].values[0]

        # Force the number of constructed dwellings for specific years if needed
        if year in [2020, 2021, 2022]:
            # Set your forced values here
            # e.g., new_dwelling_ind = 1000, new_dwelling_col = 500
            pass

        # Convert to number of dwellings
        else:
            new_dwelling_ind = effective_construction_ind / \
                               new.filter(year=year, scenario=new_scenario, variable='Individual_dwelling').data[
                                   'value'].values[0]
            new_dwelling_col = effective_construction_col / \
                               new.filter(year=year, scenario=new_scenario, variable='Collective_dwelling').data[
                                   'value'].values[0]

        new_dwelling = new_dwelling_ind + new_dwelling_col

        # Append data for each year
        for key, value in zip(data.keys(),
                              [effective_construction, effective_construction_col, effective_construction_ind,
                               new_dwelling, new_dwelling_col, new_dwelling_ind]):
            data[key].append({'year': year, 'value': value})

    # Create IamDataFrame objects for each variable
    iamdf_objects = {}
    for key in data.keys():
        df = pd.DataFrame(data[key])
        iamdf_objects[key] = pyam.IamDataFrame(df, model='CSTB', scenario=created_scenario_construction_name,
                                               region='France', variable=key.replace('_', ' ').title(),
                                               unit='Unit here')

    # Concatenate all IamDataframes into a single IamDataFrame
    combined_construction_iamdf = pyam.concat(list(iamdf_objects.values()))

    return combined_construction_iamdf


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


def calculate_cumulated_operational_ghge_new_construction(construction, operational_ghge):
    construction_col_list = []
    construction_ind_list = []
    construction_res_list = []

    for scenario in operational_ghge.scenario:
        operational_ghge_col = \
        operational_ghge.filter(scenario=scenario, variable='Collective_dwelling').data
        operational_ghge_ind = \
        operational_ghge.filter(scenario=scenario, variable='Individual_dwelling').data

        for construction_scenario in construction.scenario:
            construction_col_df = construction.filter(scenario=construction_scenario, variable='Collective_dwelling', unit='Surface in m2').data
            construction_ind_df = construction.filter(scenario=construction_scenario, variable='Individual_dwelling', unit='Surface in m2').data

            # Initialize a variable to store cumulative values
            cumulative_values_col = 0
            cumulative_values_ind = 0

            for i, year in enumerate(construction_col_df['year']):
                # Get the value for the current year
                value_col = construction_col_df['value'][i]
                value_ind = construction_ind_df['value'][i]

                value_op_col = operational_ghge_col['value'][i]
                value_op_ind = operational_ghge_ind['value'][i]

                # Update the cumulative values
                cumulative_values_col += value_col
                cumulative_values_ind += value_ind

                # Calculate the combined value by multiplying cumulative construction value with embodied_ghge value
                construction_operational_col = cumulative_values_col * value_op_col / 10e8
                construction_operational_ind = cumulative_values_ind * value_op_ind / 10e8
                construction_operational_res = construction_operational_col + construction_operational_ind

                scenario_name = f'{construction_scenario}_{scenario}'

                # Create df
                result_col_df = pd.DataFrame(
                    {'year': year,
                     'model': 'CSTB',
                     'scenario': scenario_name,
                     'region': 'France',
                     'variable': construction_col_df.variable[0],
                     'unit': 'MtCO2eq',
                     'value': construction_operational_col}, index=pd.RangeIndex(1))

                result_ind_df = pd.DataFrame(
                    {'year': year,
                     'model': 'CSTB',
                     'scenario': scenario_name,
                     'region': 'France',
                     'variable': construction_ind_df.variable[0],
                     'unit': 'MtCO2eq',
                     'value': construction_operational_ind}, index=pd.RangeIndex(1))

                result_res_df = pd.DataFrame(
                    {'year': year,
                     'model': 'CSTB',
                     'scenario': scenario_name,
                     'region': 'France',
                     'variable': 'Residential_dwelling',
                     'unit': 'MtCO2eq',
                     'value': construction_operational_res}, index=pd.RangeIndex(1))

                construction_col_list.append(result_col_df)
                construction_ind_list.append(result_ind_df)
                construction_res_list.append(result_res_df)

    combined_construction_operational_col_df = pd.concat(construction_col_list)
    combined_construction_operational_ind_df = pd.concat(construction_ind_list)
    combined_construction_operational_res_df = pd.concat(construction_res_list)

    # Concatenate both DataFrames into a single IamDataFrame
    combined_construction_operational_df = pd.concat([combined_construction_operational_col_df, combined_construction_operational_ind_df, combined_construction_operational_res_df])
    combined_construction_operational_iamdf = pyam.IamDataFrame(combined_construction_operational_df)

    return combined_construction_operational_iamdf


def calculate_nrj_new_construction(iamdf):
    # Filter out the rows with the unit 'Surface in m2'
    surface_df = iamdf.filter(unit='Surface in m2')
    surface_df = surface_df.timeseries().reset_index()

    # Get the list of years from the columns
    years = surface_df.columns[5:]

    # Initialize a list to store the results
    results = []

    # Initialize a dictionary to store the sum of collective and individual for each scenario and each year
    residential_sums = {}

    # Loop over each row in the dataframe
    for index, row in surface_df.iterrows():
        scenario = row['scenario']
        # Initialize residential sums for the scenario if not already present
        if scenario not in residential_sums:
            residential_sums[scenario] = {year: {'elec': 0, 'bois': 0, 'rcu': 0, 'value': 0} for year in years}

        # Perform calculations based on the type of dwelling
        for year in years:
            ef_nrj_entries = []  # To keep track of the ef_nrj entries for the current row
            if 'Collective' in row['variable']:
                # Calculate energy consumption in kwh ep
                ep = row[year] * 70  # 70kwhep/m2 for collective dwelling
                # Calculate the energy consumption in kwh ef for each vector
                elec = ep * 0.9 / 2.3
                rcu = ep * 0.1
                # Append energy vector entries
                ef_nrj_entries.append({'vector': 'elec', 'value': elec})
                ef_nrj_entries.append({'vector': 'rcu', 'value': rcu})
                # Add to the residential sum
                residential_sums[scenario][year]['elec'] += elec
                residential_sums[scenario][year]['rcu'] += rcu
                residential_sums[scenario][year]['value'] += elec + rcu

            elif 'Individual' in row['variable']:
                # Calculate energy consumption in kwh ep
                ep = row[year] * 55  # 55kwhep/m2 for individual dwelling
                # Calculate the energy consumption in kwh ef for each vector
                elec = ep * 0.8 / 2.3
                bois = ep * 0.2
                # Append energy vector entries
                ef_nrj_entries.append({'vector': 'elec', 'value': elec})
                ef_nrj_entries.append({'vector': 'bois', 'value': bois})
                # Add to the residential sum
                residential_sums[scenario][year]['elec'] += elec
                residential_sums[scenario][year]['bois'] += bois
                residential_sums[scenario][year]['value'] += elec + bois

            # Append the results to the results list for each energy vector
            for ef_nrj_entry in ef_nrj_entries:
                results.append({
                    'year': year,
                    'model': 'CSTB',
                    'scenario': scenario,
                    'region': 'France',
                    'variable': row['variable'],
                    'unit': 'kWh_ef',
                    'dpe': 'new',
                    'ef_nrj': ef_nrj_entry['vector'],
                    'value': ef_nrj_entry['value']
                })

    # Add residential sums to the results for each scenario
    for scenario, yearly_sums in residential_sums.items():
        for year, sums in yearly_sums.items():
            for vector, vector_value in sums.items():
                if vector == 'value':
                    continue  # Skip the total value entry
                results.append({
                    'year': year,
                    'model': 'CSTB',
                    'scenario': scenario,
                    'region': 'France',
                    'variable': 'Residential_dwelling',
                    'unit': 'kWh_ef',
                    'dpe': 'new',
                    'ef_nrj': vector,
                    'value': vector_value
                })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Create a Pyam IamDataFrame from the results DataFrame
    result_iamdf = pyam.IamDataFrame(result_df)

    return result_iamdf


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


def calculate_cumulated_combined_operational_stock_iamdf(stock, ghge):
    '''
     Calculates cumulated operational GHGE of the stock,
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
                        stock_df = stock.filter(scenario=stock_scenario, variable=stock_variable, dpe=stock_dpe,
                                                ef_nrj=ef_nrj).data
                        ghge_df = ghge.filter(scenario=ghge_scenario, variable=ef_nrj).data

                        # Check if there is data to multiply, otherwise skip to the next iteration
                        if stock_df.empty or ghge_df.empty:
                            continue

                        # Calculate GHGE in MtCO2eq
                        value_df = stock_df['value'].cumsum() * ghge_df['value'] / 10e8

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


def calculate_cumulated_nrj_new_construction(stock):
    '''
       Calculates cumulated operational GHGE of the stock,
       by multiplying kWh of energy carriers by the corresponding GHGE factors.

       Returns a unique pyam dataframe
       '''

    result_list = []

    for stock_scenario in stock.scenario:
        for stock_variable in stock.variable:
            for stock_dpe in stock.dpe:
                for ef_nrj in stock.ef_nrj:  # ['bois', 'charbon', 'elec', 'fioul', 'gaz', 'gpl', 'rcu']
                        # Make sure we're matching the energy carrier in both dataframes
                        stock_df = stock.filter(scenario=stock_scenario, variable=stock_variable, dpe=stock_dpe,
                                                ef_nrj=ef_nrj).data

                        # Check if there is data to multiply, otherwise skip to the next iteration
                        if stock_df.empty:
                            continue

                        # Calculate GHGE in MtCO2eq
                        value_df = stock_df['value'].cumsum()

                        # Construct the result dataframe
                        result_df = stock_df.copy()
                        result_df['value'] = value_df
                        result_df['scenario'] = f'{stock_scenario}'
                        result_df['variable'] = stock_variable
                        result_df['unit'] = 'kWh'

                        result_list.append(result_df)

    # Convert the results to an IamDataFrame
    results_df = pd.concat(result_list)
    results_iamdf = pyam.IamDataFrame(results_df)

    return results_iamdf



# ========================================
# Scripts
# ========================================
