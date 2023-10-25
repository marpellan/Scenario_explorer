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



# ========================================
# Internal imports
# ========================================

combined_construction_iamdf = pyam.IamDataFrame(RESULT_HOUSING_DIRECTORY_PATH / "combined_construction_iamdf.csv", sep=';')
embodied_ghge_new_construction = pyam.IamDataFrame(SCENARIOS_DIRECTORY_PATH / "ghge" / "embodied_ghge_new_construction_scenarios.csv", sep=';')


# ========================================
# Scripts
# ========================================


def monte_carlo_simulation_with_iamdf(construction_iamdf, embodied_ghge_iamdf, percentage=20, n_draws=1000):
    # Extract data from IamDataFrame
    construction_df = construction_iamdf.data
    embodied_ghge_df = embodied_ghge_iamdf.data

    years = construction_df['year'].unique()[::10]  # Selecting every 10th year
    results = []

    for year in years:
        m2_col = construction_df[(construction_df['year'] == year) & (construction_df['variable'] == 'Collective_dwelling')]['value'].values[0]
        m2_ind = construction_df[(construction_df['year'] == year) & (construction_df['variable'] == 'Individual_dwelling')]['value'].values[0]
        kgCO2eq_per_m2_col = embodied_ghge_df[(embodied_ghge_df['year'] == year) & (embodied_ghge_df['variable'] == 'Collective_dwelling')]['value'].values[0]
        kgCO2eq_per_m2_ind = embodied_ghge_df[(embodied_ghge_df['year'] == year) & (embodied_ghge_df['variable'] == 'Individual_dwelling')]['value'].values[0]

        for _ in range(n_draws):
            random_multiplier_col = 1 + np.random.uniform(-percentage/100, percentage/100)
            random_multiplier_ind = 1 + np.random.uniform(-percentage/100, percentage/100)

            total_emissions_col = m2_col * kgCO2eq_per_m2_col * random_multiplier_col
            total_emissions_ind = m2_ind * kgCO2eq_per_m2_ind * random_multiplier_ind

            results.append(total_emissions_col + total_emissions_ind)

    # Plotting the histogram
    plt.hist(results, bins=50, color='c', edgecolor='k', alpha=0.7)
    plt.title('Distribution of Total Emissions')
    plt.xlabel('Total Emissions (kgCO2eq)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# This is a mock representation of the function. Actual execution requires the data and the environment.
"Function adjusted for IamDataFrame."


t = monte_carlo_simulation_with_iamdf(combined_construction_iamdf, combined_construction_iamdf, percentage=20, n_draws=1000)