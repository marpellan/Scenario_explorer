# ========================================
# External imports
# ========================================

import matplotlib.pyplot as plt
import numpy as np
import pyam
from collections import defaultdict
from importlib import resources

# ========================================
# Internal imports
# ========================================

from trajectoire.config import data
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
RESULT_NEW_NRJ_PATH = RESULT_DIRECTORY_PATH / "results_new_nrj"



# ========================================
# Internal imports
# ========================================

combined_construction_iamdf = pyam.IamDataFrame(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_iamdf.csv", sep=';')
embodied_ghge_new_construction = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_new_construction_scenarios.csv", sep=';')

surface_reno_df = pyam.IamDataFrame(RESULT_NEW_NRJ_PATH / "combined_reno_surface_iamdf.csv", sep=';')
embodied_ghge_df = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_renovation_scenarios.csv", sep=';')

# ========================================
# Scripts
# ========================================

def run_monte_carlo_one_year(surface_df, ghge, year,
                    surface_scenario=None,
                    ghge_scenario=None,
                    iterations=100000):


    # Initialize a list to store total emissions for each iteration
    total_emissions_per_class = []

    # Get the surface data for the specific year, per saut_classe_dpe
    for saut_classe in ['1', '2', '3', '4', '5', '6']:
        surface_data_col = surface_df.filter(variable='Collective_dwelling', scenario=surface_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]
        surface_data_ind = surface_df.filter(variable='Individual_dwelling', scenario=surface_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]

        # Loop over each saut_classe_dpe in the surface dataframe
        for saut_classe in ghge.saut_classe_dpe:
            # Get the corresponding GHGE value for the specific year and saut_classe_dpe
            ghge_value_col = ghge.filter(variable='Collective_dwelling', scenario=ghge_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]
            ghge_value_ind = ghge.filter(variable='Individual_dwelling', scenario=ghge_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]

            # Define the GHGE range as +/- 25% of the GHGE value
            ghge_range_col = np.random.uniform(0.75 * ghge_value_col, 1.25 * ghge_value_ind, size=iterations)
            ghge_range_ind = np.random.uniform(0.75 * ghge_value_col, 1.25 * ghge_value_ind, size=iterations)

            # Calculate emissions for this saut_classe_dpe and add to the total
            emissions = ((ghge_range_col * surface_data_col) + (ghge_range_ind * surface_data_ind)) / 10e9
            total_emissions_per_class.append(emissions)

    # Aggregate emissions across all saut_classe_dpe
    total_emissions = np.sum(total_emissions_per_class, axis=0)

    # Calculate summary statistics
    median_value = np.median(total_emissions)
    percentile_25 = np.percentile(total_emissions, 25)
    percentile_75 = np.percentile(total_emissions, 75)

    # Plot the distribution of total emissions for the year
    plt.hist(total_emissions, bins=50, alpha=0.7, color='#22223b')
    plt.axvline(median_value, color='k', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(percentile_25, color='green', linestyle='dashed', linewidth=1, label='25th Percentile')
    plt.axvline(percentile_75, color='red', linestyle='dashed', linewidth=1, label='75th Percentile')

    plt.title(f"Distribution of renovation embodied GHGE for {year}", fontsize=16, fontweight="bold")
    plt.xlabel("MtCO2eq", fontsize=16, fontweight="bold")
    plt.ylabel("Frequency", fontsize=16, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_monte_carlo_multiple_years(surface_df, ghge,
                                   years, surface_scenario=None,
                                   ghge_scenario=None,
                                   iterations=1000000):
    # Initialize a dictionary to store total emissions for each year
    total_emissions_per_year = {}

    # Define a color for each year, ensure there are as many colors as years
    colors = ['#f8dda4', '#f9a03f', '#d45113', '#813405']
    year_colors = dict(zip(years, colors[:len(years)]))

    for year in years:
        total_emissions_per_class = []

        for saut_classe in ['1', '2', '3', '4', '5', '6']:
            surface_data_col = surface_df.filter(variable='Collective_dwelling', scenario=surface_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]
            surface_data_ind = surface_df.filter(variable='Individual_dwelling', scenario=surface_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]

            # Loop over each saut_classe_dpe in the surface dataframe
            for saut_classe in ghge.saut_classe_dpe:
                ghge_value_col = ghge.filter(variable='Collective_dwelling', scenario=ghge_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]
                ghge_value_ind = ghge.filter(variable='Individual_dwelling', scenario=ghge_scenario, saut_classe_dpe=saut_classe, year=year).data['value'][0]

                ghge_range_col = np.random.uniform(0.75 * ghge_value_col, 1.25 * ghge_value_col, size=iterations)
                ghge_range_ind = np.random.uniform(0.75 * ghge_value_ind, 1.25 * ghge_value_ind, size=iterations)

                emissions = ((ghge_range_col * surface_data_col) + (ghge_range_ind * surface_data_ind)) / 10e9
                total_emissions_per_class.append(emissions)

        total_emissions = np.sum(total_emissions_per_class, axis=0)
        total_emissions_per_year[year] = total_emissions

    # Plot the distribution of total emissions for all years
    plt.figure(figsize=(10, 6))

    for year, emissions in total_emissions_per_year.items():
        median_value = np.median(emissions)
        percentile_25 = np.percentile(emissions, 25)
        percentile_75 = np.percentile(emissions, 75)

        # Plot histogram
        plt.hist(emissions, bins=50, alpha=0.7, label=f'{year}, {median_value:.2f}MtCO2eq (median)', color=year_colors[year])

        # Plot the median and percentiles for each year
        plt.axvline(median_value, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(percentile_25, color='green', linestyle='dashed', linewidth=1)
        plt.axvline(percentile_75, color='red', linestyle='dashed', linewidth=1)

    plt.title(f"Embodied renovation GHGE - {surface_scenario} & {ghge_scenario}", fontsize=16, fontweight="bold")
    plt.xlabel("MtCO2eq", fontsize=16, fontweight="bold")
    plt.ylabel("Frequency", fontsize=16, fontweight="bold")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.xlim(0, 20)
    plt.tight_layout()
    plt.show()


# Example usage:
# run_monte_carlo(surface_df, ghge, years=[2020, 2030, 2040], surface_scenario='HRS_linear', ghge_scenario='BAU_RE2020_WLC')


# Run the Monte Carlo simulation for the year 2050
# fig_aps = run_monte_carlo_one_year(surface_reno_df,
#                           embodied_ghge_df,
#                           surface_scenario='HRS_linear',
#                           ghge_scenario='BAU_RE2020_WLC',
#                           year=2050)


# High renovation scenarios
fig_hrs_linear_re2020 = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='HRS_linear',
                          ghge_scenario='BAU_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])


fig_hrs_linear_re2020_optimist = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='HRS_linear',
                          ghge_scenario='Optimist_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])

fig_hrs_plateau_re2020 = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='HRS_plateau',
                          ghge_scenario='BAU_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])


fig_hrs_plateau_re2020_optimist = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='HRS_plateau',
                          ghge_scenario='Optimist_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])

# Medium renovation scenarios
fig_mrs_linear_re2020 = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='MRS_linear',
                          ghge_scenario='BAU_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])


fig_mrs_linear_re2020_optimist = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='MRS_linear',
                          ghge_scenario='Optimist_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])

fig_mrs_plateau_re2020 = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='MRS_plateau',
                          ghge_scenario='BAU_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])


fig_mrs_plateau_re2020_optimist = run_monte_carlo_multiple_years(surface_reno_df,
                          embodied_ghge_df,
                          surface_scenario='MRS_plateau',
                          ghge_scenario='Optimist_RE2020_WLC',
                          years=[2020, 2030, 2040, 2050])


