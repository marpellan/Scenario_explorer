# -*- coding: utf-8 -*-

# ========================================
# External imports
# ========================================

import pandas as pd

from importlib import resources

# ========================================
# External CSTB imports
# ========================================


# ========================================
# Internal imports
# ========================================

from trajectoire.core.model.generate_distribution_scenarios import (
    generate_distribution,
    plot_distributions,
)
from trajectoire import results

# ========================================
# Constants
# ========================================

RESULT_DIRECTORY_PATH = resources.files(results)

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

# === INPUT DATA ===

### Total number of renovations in the HRS
potential_surface_hrs = 2241461310
potential_surface_col_hrs = 700315156
potential_surface_ind_hrs = 1541146154

potential_dwelling_hrs = 26996501
potential_dwelling_col_hrs = 11943267
potential_dwelling_ind_hrs = 15053234

### Total number of renovations in the MRS
potential_surface_mrs = 925010282
potential_surface_col_mrs = 306509555
potential_surface_ind_mrs = 618500727

potential_dwelling_mrs = 11632613
potential_dwelling_col_mrs = 5004234
potential_dwelling_ind_mrs = 6628379

# Number of years for which you want to generate the distributions
start_year = 2020
num_years = 31

# === CREATE DF FOR EACH DWELLING TYPE IN NUMBER OF DWELLING AND SURFACE FOR HIGH RENOVATION SCENARIOS
## 1) FOR LC (dwellings)
### Create DataFrames for each scenario
df_constant_dwelling_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_hrs, num_years, 'constant')
})

df_exponential_dwelling_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_hrs, num_years, 'exponential')
})

df_exponential_slow_dwelling_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_hrs, num_years, 'slow_exponential')
})

df_normal_dwelling_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_hrs, num_years, 'normal', mu=num_years/2, sigma=num_years/6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_dwelling_lc_hrs['Renovations'] = df_exponential_dwelling_lc_hrs['Renovations'] * potential_dwelling_col_hrs / df_exponential_dwelling_lc_hrs['Renovations'].sum()
df_exponential_slow_dwelling_lc_hrs['Renovations'] = df_exponential_slow_dwelling_lc_hrs['Renovations'] * potential_dwelling_col_hrs / df_exponential_slow_dwelling_lc_hrs['Renovations'].sum()
df_normal_dwelling_lc_hrs['Renovations'] = df_normal_dwelling_lc_hrs['Renovations'] * potential_dwelling_col_hrs / df_normal_dwelling_lc_hrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_dwelling_lc_hrs, 'Constant Distribution'),
    (df_exponential_dwelling_lc_hrs, 'Exponential Distribution'),
    (df_exponential_slow_dwelling_lc_hrs, 'Exponential Distribution with slow rise'),
    (df_normal_dwelling_lc_hrs, 'Normal Distribution'),

    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_dwelling_lc_hrs.xlsx") as writer:
    df_constant_dwelling_lc_hrs.to_excel(writer, sheet_name='constant')
    df_exponential_dwelling_lc_hrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_dwelling_lc_hrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_dwelling_lc_hrs.to_excel(writer, sheet_name='normal')


## 2) FOR LC (surface)
### Create DataFrames for each scenario
df_constant_surface_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_hrs, num_years, 'constant')
})

df_exponential_surface_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_hrs, num_years, 'exponential')
})

df_exponential_slow_surface_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_hrs, num_years, 'slow_exponential')
})
df_normal_surface_lc_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_hrs, num_years, 'normal', mu=num_years/2, sigma=num_years/6)
})


### Normalize the values to make sure they sum up to total_renovations
df_exponential_surface_lc_hrs['Renovations'] = df_exponential_surface_lc_hrs['Renovations'] * potential_surface_col_hrs / df_exponential_surface_lc_hrs['Renovations'].sum()
df_exponential_slow_surface_lc_hrs['Renovations'] = df_exponential_slow_surface_lc_hrs['Renovations'] * potential_surface_col_hrs / df_exponential_slow_surface_lc_hrs['Renovations'].sum()
df_normal_surface_lc_hrs['Renovations'] = df_normal_surface_lc_hrs['Renovations'] * potential_surface_col_hrs / df_normal_surface_lc_hrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_surface_lc_hrs, 'Constant Distribution'),
    (df_exponential_surface_lc_hrs, 'Exponential Distribution'),
    (df_exponential_slow_surface_lc_hrs, 'Exponential Distribution with slow rise'),
    (df_normal_surface_lc_hrs, 'Normal distribution'),
    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_surface_lc_hrs.xlsx") as writer:
    df_constant_surface_lc_hrs.to_excel(writer, sheet_name='constant')
    df_exponential_surface_lc_hrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_surface_lc_hrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_surface_lc_hrs.to_excel(writer, sheet_name='normal')


## 3) FOR MI (dwellings)
### Create DataFrames for each scenario
df_constant_dwelling_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_hrs, num_years, 'constant')
})

df_exponential_dwelling_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_hrs, num_years, 'exponential')
})

df_exponential_slow_dwelling_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_hrs, num_years, 'slow_exponential')
})

df_normal_dwelling_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_hrs, num_years, 'normal', mu=num_years/2, sigma=num_years/6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_dwelling_mi_hrs['Renovations'] = df_exponential_dwelling_mi_hrs['Renovations'] * potential_dwelling_ind_hrs / df_exponential_dwelling_mi_hrs['Renovations'].sum()
df_exponential_slow_dwelling_mi_hrs['Renovations'] = df_exponential_slow_dwelling_mi_hrs['Renovations'] * potential_dwelling_ind_hrs / df_exponential_slow_dwelling_mi_hrs['Renovations'].sum()
df_normal_dwelling_mi_hrs['Renovations'] = df_normal_dwelling_mi_hrs['Renovations'] * potential_dwelling_ind_hrs / df_normal_dwelling_mi_hrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_dwelling_mi_hrs, 'Constant Distribution'),
    (df_exponential_dwelling_mi_hrs, 'Exponential Distribution'),
    (df_exponential_slow_dwelling_mi_hrs, 'Exponential Distribution with slow rise'),
    (df_normal_dwelling_mi_hrs, 'Normal distribution'),
    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_dwelling_mi_hrs.xlsx") as writer:
    df_constant_dwelling_mi_hrs.to_excel(writer, sheet_name='constant')
    df_exponential_dwelling_mi_hrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_dwelling_mi_hrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_dwelling_mi_hrs.to_excel(writer, sheet_name='normal')



## 4) FOR MI (surface)
### Create DataFrames for each scenario
df_constant_surface_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_hrs, num_years, 'constant')
})

df_exponential_surface_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_hrs, num_years, 'exponential')
})

df_exponential_slow_surface_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_hrs, num_years, 'slow_exponential')
})

df_normal_surface_mi_hrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_hrs, num_years, 'normal', mu=num_years/2, sigma=num_years/6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_surface_mi_hrs['Renovations'] = df_exponential_surface_mi_hrs['Renovations'] * potential_surface_ind_hrs / df_exponential_surface_mi_hrs['Renovations'].sum()
df_exponential_slow_surface_mi_hrs['Renovations'] = df_exponential_slow_surface_mi_hrs['Renovations'] * potential_surface_ind_hrs / df_exponential_slow_surface_mi_hrs['Renovations'].sum()
df_normal_surface_mi_hrs['Renovations'] = df_normal_surface_mi_hrs['Renovations'] * potential_surface_ind_hrs / df_normal_surface_mi_hrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_surface_mi_hrs, 'Constant Distribution'),
    (df_exponential_surface_mi_hrs, 'Exponential Distribution'),
    (df_exponential_slow_surface_mi_hrs, 'Exponential Distribution with slow rise'),
    (df_normal_surface_mi_hrs, 'Normal distribution'),
    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_surface_mi_hrs.xlsx") as writer:
    df_constant_surface_mi_hrs.to_excel(writer, sheet_name='constant')
    df_exponential_surface_mi_hrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_surface_mi_hrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_surface_mi_hrs.to_excel(writer, sheet_name='normal')

# === CREATE DF FOR EACH DWELLING TYPE IN NUMBER OF DWELLING AND SURFACE FOR MEDIUM RENOVATION SCENARIOS
## 1) FOR LC (dwellings)
### Create DataFrames for each scenario
df_constant_dwelling_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_mrs, num_years, 'constant')
})

df_exponential_dwelling_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_mrs, num_years, 'exponential')
})

df_exponential_slow_dwelling_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_mrs, num_years, 'slow_exponential')
})

df_normal_dwelling_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_col_mrs, num_years, 'normal', mu=num_years / 2,
                                         sigma=num_years / 6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_dwelling_lc_mrs['Renovations'] = df_exponential_dwelling_lc_mrs[
                                                    'Renovations'] * potential_dwelling_col_mrs / \
                                                df_exponential_dwelling_lc_mrs['Renovations'].sum()
df_exponential_slow_dwelling_lc_mrs['Renovations'] = df_exponential_slow_dwelling_lc_mrs[
                                                         'Renovations'] * potential_dwelling_col_mrs / \
                                                     df_exponential_slow_dwelling_lc_mrs['Renovations'].sum()
df_normal_dwelling_lc_mrs['Renovations'] = df_normal_dwelling_lc_mrs['Renovations'] * potential_dwelling_col_mrs / \
                                           df_normal_dwelling_lc_mrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_dwelling_lc_mrs, 'Constant Distribution'),
    (df_exponential_dwelling_lc_mrs, 'Exponential Distribution'),
    (df_exponential_slow_dwelling_lc_mrs, 'Exponential Distribution with slow rise'),
    (df_normal_dwelling_lc_mrs, 'Normal Distribution'),

title_size = 16,  # Set custom title font size
             legend_size = 12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_dwelling_lc_mrs.xlsx") as writer:
    df_constant_dwelling_lc_mrs.to_excel(writer, sheet_name='constant')
    df_exponential_dwelling_lc_mrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_dwelling_lc_mrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_dwelling_lc_mrs.to_excel(writer, sheet_name='normal')

## 2) FOR LC (surface)
### Create DataFrames for each scenario
df_constant_surface_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_mrs, num_years, 'constant')
})

df_exponential_surface_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_mrs, num_years, 'exponential')
})

df_exponential_slow_surface_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_mrs, num_years, 'slow_exponential')
})
df_normal_surface_lc_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_col_mrs, num_years, 'normal', mu=num_years / 2,
                                         sigma=num_years / 6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_surface_lc_mrs['Renovations'] = df_exponential_surface_lc_mrs[
                                                   'Renovations'] * potential_surface_col_mrs / \
                                               df_exponential_surface_lc_mrs['Renovations'].sum()
df_exponential_slow_surface_lc_mrs['Renovations'] = df_exponential_slow_surface_lc_mrs[
                                                        'Renovations'] * potential_surface_col_mrs / \
                                                    df_exponential_slow_surface_lc_mrs['Renovations'].sum()
df_normal_surface_lc_mrs['Renovations'] = df_normal_surface_lc_mrs['Renovations'] * potential_surface_col_mrs / \
                                          df_normal_surface_lc_mrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_surface_lc_mrs, 'Constant Distribution'),
    (df_exponential_surface_lc_mrs, 'Exponential Distribution'),
    (df_exponential_slow_surface_lc_mrs, 'Exponential Distribution with slow rise'),
    (df_normal_surface_lc_mrs, 'Normal distribution'),
    title_size = 16,  # Set custom title font size
    legend_size = 12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_surface_lc_mrs.xlsx") as writer:
    df_constant_surface_lc_mrs.to_excel(writer, sheet_name='constant')
    df_exponential_surface_lc_mrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_surface_lc_mrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_surface_lc_mrs.to_excel(writer, sheet_name='normal')

## 3) FOR MI (dwellings)
### Create DataFrames for each scenario
df_constant_dwelling_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_mrs, num_years, 'constant')
})

df_exponential_dwelling_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_mrs, num_years, 'exponential')
})

df_exponential_slow_dwelling_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_mrs, num_years, 'slow_exponential')
})

df_normal_dwelling_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_dwelling_ind_mrs, num_years, 'normal', mu=num_years / 2,
                                         sigma=num_years / 6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_dwelling_mi_mrs['Renovations'] = df_exponential_dwelling_mi_mrs[
                                                    'Renovations'] * potential_dwelling_ind_mrs / \
                                                df_exponential_dwelling_mi_mrs['Renovations'].sum()
df_exponential_slow_dwelling_mi_mrs['Renovations'] = df_exponential_slow_dwelling_mi_mrs[
                                                         'Renovations'] * potential_dwelling_ind_mrs / \
                                                     df_exponential_slow_dwelling_mi_mrs['Renovations'].sum()
df_normal_dwelling_mi_mrs['Renovations'] = df_normal_dwelling_mi_mrs['Renovations'] * potential_dwelling_ind_mrs / \
                                           df_normal_dwelling_mi_mrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_dwelling_mi_mrs, 'Constant Distribution'),
    (df_exponential_dwelling_mi_mrs, 'Exponential Distribution'),
    (df_exponential_slow_dwelling_mi_mrs, 'Exponential Distribution with slow rise'),
    (df_normal_dwelling_mi_mrs, 'Normal distribution'),
    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_dwelling_mi_mrs.xlsx") as writer:
    df_constant_dwelling_mi_mrs.to_excel(writer, sheet_name='constant')
    df_exponential_dwelling_mi_mrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_dwelling_mi_mrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_dwelling_mi_mrs.to_excel(writer, sheet_name='normal')

## 4) FOR MI (surface)
### Create DataFrames for each scenario
df_constant_surface_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_mrs, num_years, 'constant')
})

df_exponential_surface_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_mrs, num_years, 'exponential')
})

df_exponential_slow_surface_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_mrs, num_years, 'slow_exponential')
})

df_normal_surface_mi_mrs = pd.DataFrame({
    'Year': range(start_year, start_year + num_years),
    'Renovations': generate_distribution(potential_surface_ind_mrs, num_years, 'normal', mu=num_years / 2,
                                         sigma=num_years / 6)
})

### Normalize the values to make sure they sum up to total_renovations
df_exponential_surface_mi_mrs['Renovations'] = df_exponential_surface_mi_mrs[
                                                   'Renovations'] * potential_surface_ind_mrs / \
                                               df_exponential_surface_mi_mrs['Renovations'].sum()
df_exponential_slow_surface_mi_mrs['Renovations'] = df_exponential_slow_surface_mi_mrs[
                                                        'Renovations'] * potential_surface_ind_mrs / \
                                                    df_exponential_slow_surface_mi_mrs['Renovations'].sum()
df_normal_surface_mi_mrs['Renovations'] = df_normal_surface_mi_mrs['Renovations'] * potential_surface_ind_mrs / \
                                          df_normal_surface_mi_mrs['Renovations'].sum()
### Plot the distributions with custom title and legend sizes
plot_distributions(
    (df_constant_surface_mi_mrs, 'Constant Distribution'),
    (df_exponential_surface_mi_mrs, 'Exponential Distribution'),
    (df_exponential_slow_surface_mi_mrs, 'Exponential Distribution with slow rise'),
    (df_normal_surface_mi_mrs, 'Normal distribution'),
    title_size=16,  # Set custom title font size
    legend_size=12  # Set custom legend font size
)

with pd.ExcelWriter(RESULT_DIRECTORY_PATH / "renovation_scenarios_surface_mi_mrs.xlsx") as writer:
    df_constant_surface_mi_mrs.to_excel(writer, sheet_name='constant')
    df_exponential_surface_mi_mrs.to_excel(writer, sheet_name='exponential_declining')
    df_exponential_slow_surface_mi_mrs.to_excel(writer, sheet_name='exponential_slow')
    df_normal_surface_mi_mrs.to_excel(writer, sheet_name='normal')
