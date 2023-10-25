import pandas as pd
import numpy as np
import pyam
import matplotlib.pyplot as plt

dpe_ademe = {
    "A": '#368062',
    "B": '#529952',
    "C": '#76a174',
    "D": '#dacf37',
    "E": '#cfa12e',
    "F": '#c4743c',
    "G": '#ae251b',
}

dpe_bdnb = {
    "A": '#3F3F3D',
    "B": '#5B755D',
    "C": '#8CA188',
    "D": '#E1BA75',
    "E": '#EE892F',
    "F": '#E76020',
    "G": '#ae251b',
}

def plot_scenarios_construction(iamdf, variable=None, unit=None, title=None, figsize=None, ylabel=None):
    plt.figure(figsize=figsize)

    data = iamdf.filter(variable=variable, unit=unit)
    plot = data.plot(color='scenario')
    plot.set_title('')

    plt.legend(loc='upper right', fontsize=12)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel(ylabel, fontsize=16, fontweight="bold")

    # plt.tight_layout()  # Ensures that the elements fit nicely in the figure area
    plt.show()


def plot_reno_surface_scenarios(iamdf, scenarios_variables, titles, figsize=None, ncols=2):
    num_plots = len(scenarios_variables)
    nrows = (num_plots + ncols - 1) // ncols  # Calculate the number of rows for subplots

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)

    for (scenario, variable, saut_classe_dpe), title, ax in zip(scenarios_variables, titles, axes.flat):
        data = iamdf.filter(scenario=scenario, variable=variable, saut_classe_dpe=saut_classe_dpe)
        plot = data.plot(ax=ax, color='scenario')
        plot.set_title('')

        # Adjust legend placement and font size
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=10)
        ax.set_title(title, fontsize=20, fontweight="bold")
        # ax.set_xlabel("Year", fontsize=10, fontweight="bold")
        # ax.set_ylabel("MtCO2eq", fontsize=10, fontweight="bold")

    plt.tight_layout()  # Adjusts spacing between subplots
    plt.show()


def plot_reno_embodied_ghge_scenarios(iamdf, scenario=None, variable=None, saut_classe_dpe=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    data = iamdf.filter(scenario=scenario, variable=variable, saut_classe_dpe=saut_classe_dpe)
    plot = data.plot(color='scenario')
    plot.set_title('')

    plt.legend(loc='upper left', ncol=1, fontsize=12)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 25)

    # plt.tight_layout()  # Ensures that the elements fit nicely in the figure area
    plt.show()


def plot_stock_surface_scenarios(iamdf, scenarios_variables, titles, figsize=None, ncols=2):
    num_plots = len(scenarios_variables)
    nrows = (num_plots + ncols - 1) // ncols  # Calculate the number of rows for subplots

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)

    for (scenario, variable, dpe_avant), title, ax in zip(scenarios_variables, titles, axes.flat):
        data = iamdf.filter(scenario=scenario, variable=variable, dpe_avant=dpe_avant)
        plot = data.plot(ax=ax, color='scenario')
        plot.set_title('')

        # Adjust legend placement and font size
        ax.legend(loc='upper left', ncol=1, fontsize=10)
        ax.set_title(title, fontsize=20, fontweight="bold")
        # ax.set_xlabel("Year", fontsize=10, fontweight="bold")
        # ax.set_ylabel("MtCO2eq", fontsize=10, fontweight="bold")

    plt.tight_layout()  # Adjusts spacing between subplots
    plt.show()


def plot_stock_operational_ghge_scenarios(iamdf, scenario=None, variable=None, dpe=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    data = iamdf.filter(scenario=scenario, variable=variable, dpe=dpe)
    plot = data.plot(color='scenario', cmap='coolwarm')
    plot.set_title('')

    plt.legend(loc='lower left', ncol=1, fontsize=12)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")


    # plt.tight_layout()  # Ensures that the elements fit nicely in the figure area
    plt.show()


# GOOD FUNCTIONS FOR EXPLORATION
def plot_cumulative_ghge_lines(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario
    unique_scenarios = filtered_data['scenario'].unique()
    for scenario in unique_scenarios:
        scenario_data = filtered_data.filter(scenario=scenario)
        cumulative_sum = scenario_data['value'].cumsum()
        plt.plot(scenario_data['year'], cumulative_sum, label=scenario)

    plt.legend(loc='upper left', fontsize=10)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_cumulative_ghge_area(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario
    unique_scenarios = filtered_data['scenario'].unique()
    for scenario in unique_scenarios:
        scenario_data = filtered_data.filter(scenario=scenario)
        cumulative_sum = scenario_data['value'].cumsum()
        plt.fill_between(scenario_data['year'], 0, cumulative_sum, label=scenario, alpha=0.5)

    plt.legend(loc='upper left', fontsize=10)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 500)


    plt.tight_layout()
    plt.show()


# GOOD FUNCTIONS FOR A PARTICULAR SCENARIO
def plot_reno_saut(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Get unique scenarios and saut_classe_dpe values
    unique_scenarios = filtered_data['scenario'].unique()
    unique_saut_classe_dpe = [1, 2, 3, 4, 5, 6]
    years = filtered_data['year'].unique()

    # Create a dictionary to store data for each scenario and saut_classe_dpe combination
    saut_classe_dpe_data = {(scenario, saut_classe_dpe): [] for scenario in unique_scenarios for saut_classe_dpe in
                            unique_saut_classe_dpe}

    for scenario in unique_scenarios:
        for saut_classe_dpe in unique_saut_classe_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, saut_classe_dpe=saut_classe_dpe)
            values = scenario_data['value']
            saut_classe_dpe_data[(scenario, saut_classe_dpe)] = values.tolist()

    # Define custom colors for each saut_classe_dpe value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, saut_classe_dpe in enumerate(unique_saut_classe_dpe):
            label = f"{scenario} - DPE {saut_classe_dpe}"
            plt.bar(years, saut_classe_dpe_data[(scenario, saut_classe_dpe)], bottom=bottom, label=label,
                    color=custom_colors[idx])
            if bottom is None:
                bottom = saut_classe_dpe_data[(scenario, saut_classe_dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, saut_classe_dpe_data[(scenario, saut_classe_dpe)])]

    plt.legend(loc='upper right', fontsize=12)
    plt.title(title, fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq or m2", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_cumulative_reno_saut(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario and saut_classe_dpe combination
    unique_scenarios = filtered_data['scenario'].unique()
    unique_saut_classe_dpe = [1, 2, 3, 4, 5, 6]
    years = filtered_data['year'].unique()

    # Create a dictionary to store cumulative sum data for each scenario and saut_classe_dpe combination
    cumulative_data = {(scenario, saut_classe_dpe): [] for scenario in unique_scenarios for saut_classe_dpe in
                       unique_saut_classe_dpe}

    for scenario in unique_scenarios:
        for saut_classe_dpe in unique_saut_classe_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, saut_classe_dpe=saut_classe_dpe)
            cumulative_sum = scenario_data['value'].cumsum()
            cumulative_data[(scenario, saut_classe_dpe)] = cumulative_sum.tolist()

    # Define custom colors for each saut_classe_dpe value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, saut_classe_dpe in enumerate(unique_saut_classe_dpe):
            label = f"{scenario} - Saut Classe DPE {saut_classe_dpe}"
            plt.bar(years, cumulative_data[(scenario, saut_classe_dpe)], bottom=bottom, label=label,
                    color=custom_colors[idx])
            if bottom is None:
                bottom = cumulative_data[(scenario, saut_classe_dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, cumulative_data[(scenario, saut_classe_dpe)])]

    plt.legend(loc='upper left', fontsize=10)
    plt.title(title, fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("Cumulative MtCO2eq or m2", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_operational_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario and dpe_avant combination
    unique_scenarios = filtered_data['scenario'].unique()
    unique_dpe = [1, 2, 3, 4, 5, 6, 7]
    years = filtered_data['year'].unique()

    # Create a dictionary to store cumulative sum data for each scenario and dpe_avant combination
    cumulative_data = {(scenario, dpe): [] for scenario in unique_scenarios for dpe in unique_dpe}

    for scenario in unique_scenarios:
        for dpe in unique_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, dpe=dpe)
            cumulative_sum = scenario_data['value']
            cumulative_data[(scenario, dpe)] = cumulative_sum.tolist()

    # Define custom colors for each DPE value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020', '#ae251b']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, dpe in enumerate(unique_dpe):
            label = f"{dpe}"
            plt.bar(years, cumulative_data[(scenario, dpe)], bottom=bottom, label=label, color=custom_colors[idx])
            if bottom is None:
                bottom = cumulative_data[(scenario, dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, cumulative_data[(scenario, dpe)])]

    plt.legend(loc='upper right', fontsize=12)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_cumulative_operational_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario and dpe_avant combination
    unique_scenarios = filtered_data['scenario'].unique()
    unique_dpe = [1, 2, 3, 4, 5, 6, 7]
    years = filtered_data['year'].unique()

    # Create a dictionary to store cumulative sum data for each scenario and dpe_avant combination
    cumulative_data = {(scenario, dpe): [] for scenario in unique_scenarios for dpe in unique_dpe}

    for scenario in unique_scenarios:
        for dpe in unique_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, dpe=dpe)
            cumulative_sum = scenario_data['value'].cumsum()
            cumulative_data[(scenario, dpe)] = cumulative_sum.tolist()

    # Define custom colors for each DPE value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020', '#ae251b']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, dpe in enumerate(unique_dpe):
            label = f"{dpe}"
            plt.bar(years, cumulative_data[(scenario, dpe)], bottom=bottom, label=label, color=custom_colors[idx])
            if bottom is None:
                bottom = cumulative_data[(scenario, dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, cumulative_data[(scenario, dpe)])]

    plt.legend(loc='upper left', fontsize=10)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_embodied_all_stock(df, title=None):
    # Filter df by programmation
    df_construction = df.filter(programmation='New_construction').data
    df_renovation = df.filter(programmation='Renovation').data
    df_demolition = df.filter(programmation='Demolition').data

    # Custom colors for each programmation
    construction_color = '#e07a5f'
    renovation_color = '#f4f1de'
    demolition_color = '#3d405b'

    years = df_construction['year']
    construction_values = df_construction['value']
    renovation_values = df_renovation['value']
    demolition_values = df_demolition['value']

    plt.figure(figsize=(10, 8))

    # Plot stacked bar chart
    plt.bar(years, construction_values, label='New Construction', color=construction_color)
    plt.bar(years, renovation_values, label='Renovation', bottom=construction_values, color=renovation_color)
    plt.bar(years, demolition_values, label='Demolition', bottom=construction_values + renovation_values,
            color=demolition_color)

    # Matplotlib setting
    plt.legend(loc='upper right', fontsize=12)
    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 20)

    plt.tight_layout()
    plt.show()


def plot_cumulative_all_stock_embodied(iamdf, title=None):
    df_construction = iamdf.filter(programmation='New_construction').data
    df_renovation = iamdf.filter(programmation='Renovation').data
    df_demolition = iamdf.filter(programmation='Demolition').data

    df_construction['value'] = df_construction['value'].cumsum()
    df_renovation['value'] = df_renovation['value'].cumsum()
    df_demolition['value'] = df_demolition['value'].cumsum()

    # Custom colors for each programmation
    construction_color = '#e07a5f'
    renovation_color = '#f4f1de'
    demolition_color = '#3d405b'

    years = df_construction['year']
    construction_values = df_construction['value']
    renovation_values = df_renovation['value']
    demolition_values = df_demolition['value']

    plt.figure(figsize=(10, 8))

    # Plot stacked bar chart
    plt.bar(years, construction_values, label='New Construction', color=construction_color)
    plt.bar(years, renovation_values, label='Renovation', bottom=construction_values, color=renovation_color)
    plt.bar(years, demolition_values, label='Demolition', bottom=construction_values + renovation_values,
            color=demolition_color)

    # Matplotlib setting
    plt.legend(loc='upper right', fontsize=12)
    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 500)

    plt.tight_layout()
    plt.show()


def plot_operational_all_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario and dpe_avant combination
    unique_scenarios = filtered_data['scenario'].unique()
    unique_dpe = ['1', '2', '3', '4', '5', '6', '7', 'new']
    #unique_dpe = [str(dpe) for dpe in unique_dpe]
    years = filtered_data['year'].unique()

    # Create a dictionary to store cumulative sum data for each scenario and dpe_avant combination
    cumulative_data = {(scenario, dpe): [] for scenario in unique_scenarios for dpe in unique_dpe}

    for scenario in unique_scenarios:
        for dpe in unique_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, dpe=dpe)
            cumulative_sum = scenario_data['value']
            cumulative_data[(scenario, dpe)] = cumulative_sum.tolist()

    # Define custom colors for each DPE value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020', '#ae251b', '#cdd7d6']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, dpe in enumerate(unique_dpe):
            label = f"{dpe}"
            plt.bar(years, cumulative_data[(scenario, dpe)], bottom=bottom, label=label, color=custom_colors[idx])
            if bottom is None:
                bottom = cumulative_data[(scenario, dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, cumulative_data[(scenario, dpe)])]

    plt.legend(loc='upper right', fontsize=12)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    # Set the y-axis range to 0-60
    plt.ylim(0, 60)

    plt.tight_layout()
    plt.show()


def plot_cumulative_operational_all_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None):
    plt.figure(figsize=figsize)

    # Filter the data based on scenario pattern and variable
    filtered_data = iamdf.filter(scenario=scenario_pattern, variable=variable)

    # Calculate the cumulative sum for each scenario and dpe_avant combination
    unique_scenarios = filtered_data['scenario'].unique()
    unique_dpe = ['1', '2', '3', '4', '5', '6', '7', 'new']
    years = filtered_data['year'].unique()

    # Create a dictionary to store cumulative sum data for each scenario and dpe_avant combination
    cumulative_data = {(scenario, dpe): [] for scenario in unique_scenarios for dpe in unique_dpe}

    for scenario in unique_scenarios:
        for dpe in unique_dpe:
            scenario_data = filtered_data.filter(scenario=scenario, dpe=dpe)
            cumulative_sum = scenario_data['value'].cumsum()
            cumulative_data[(scenario, dpe)] = cumulative_sum.tolist()

    # Define custom colors for each DPE value
    custom_colors = ['#3F3F3D', '#5B755D', '#8CA188', '#E1BA75', '#EE892F', '#E76020', '#ae251b', '#cdd7d6']

    # Create a stacked bar plot with custom colors
    bottom = None
    for scenario in unique_scenarios:
        for idx, dpe in enumerate(unique_dpe):
            label = f"{dpe}"
            plt.bar(years, cumulative_data[(scenario, dpe)], bottom=bottom, label=label, color=custom_colors[idx])
            if bottom is None:
                bottom = cumulative_data[(scenario, dpe)]
            else:
                bottom = [b + c for b, c in zip(bottom, cumulative_data[(scenario, dpe)])]

    plt.legend(loc='upper left', fontsize=10)

    plt.title(title, fontsize=20, fontweight="bold")

    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    # Set the y-axis range
    plt.ylim(0, 1600)

    plt.tight_layout()
    plt.show()


def wlc_graph(operational_bau, operational_optimistic, embodied_bau, embodied_optimistic, year=None):
    # Calculate the operational and embodied values for each scenario
    operational_bau_value = operational_bau.filter(year=year, variable='Residential_dwelling').data['value'].sum()
    operational_optimistic_value = operational_optimistic.filter(year=year, variable='Residential_dwelling').data[
        'value'].sum()
    embodied_bau_value = embodied_bau.filter(year=year, variable='Residential_dwelling').data['value'].sum()
    embodied_optimistic_value = embodied_optimistic.filter(year=year, variable='Residential_dwelling').data[
        'value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = ['BAU', 'Sufficiency']
    operational_values = [operational_bau_value, operational_optimistic_value]
    embodied_values = [embodied_bau_value, embodied_optimistic_value]

    plt.figure(figsize=(10, 8))
    plt.bar(scenarios, operational_values, label='Operational', color='#FFBA08')
    plt.bar(scenarios, embodied_values, label='Embodied', bottom=operational_values, color='#1C3144')
    #plt.xticks(rotation=45)
    plt.xticks(fontsize=14, fontweight='bold')
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(f'Whole life GHGE in {year}', fontsize=20, fontweight='bold')

    # Add horizontal lines at specific y-values
    plt.axhline(y=7, color='lightgray', linestyle='--', label='SNBC operational')
    plt.axhline(y=10, color='darkgray', linestyle='--', label='SDS embodied')
    plt.axhline(y=17, color='black', linestyle='--', label='SNBC + SDS')

    plt.legend()
    plt.show()

def wlc_cum_graph(operational_bau, operational_optimistic, embodied_bau, embodied_optimistic):
    # Calculate the operational and embodied values for each scenario
    operational_bau_value = operational_bau.filter(variable='Residential_dwelling').data['value'].sum()
    operational_optimistic_value = operational_optimistic.filter(variable='Residential_dwelling').data['value'].sum()
    embodied_bau_value = embodied_bau.filter(variable='Residential_dwelling').data['value'].sum()
    embodied_optimistic_value = embodied_optimistic.filter(variable='Residential_dwelling').data['value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = ['BAU', 'Sufficiency']
    operational_values = [operational_bau_value, operational_optimistic_value]
    embodied_values = [embodied_bau_value, embodied_optimistic_value]

    plt.figure(figsize=(10, 8))
    plt.bar(scenarios, operational_values, label='Operational', color='#FFBA08')
    plt.bar(scenarios, embodied_values, label='Embodied', bottom=operational_values, color='#1C3144')
    #plt.xticks(rotation=45)
    plt.xticks(fontsize=14, fontweight='bold')
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(f'Cumulated whole life GHGE', fontsize=20, fontweight='bold')

    # Add horizontal lines at specific y-values
    # plt.axhline(y=17, color='black', linestyle='--', label='SNBC + SDS')

    plt.legend()
    plt.show()

def wlc_graph_sensibility(combined_operational_iamdf, combined_embodied_iamdf, year=None):
    # Filter operational scenarios with extreme cases
    ## BAU scenarios
    operational_bau_mrs_constant = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                     scenario='BAU_construction_Constant_MRS_rising_HDS_Constant').data[
        'value'].sum()
    operational_bau_hrs_constant = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                     scenario='BAU_construction_Constant_HRS_rising_HDS_Constant').data[
        'value'].sum()
    operational_bau_mrs_half = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                 scenario='BAU_construction_Half_MRS_rising_HDS_Half').data[
        'value'].sum()
    operational_bau_hrs_half = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                 scenario='BAU_construction_Half_HRS_rising_HDS_Half').data[
        'value'].sum()
    operational_bau_mrs_quarter = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                    scenario='BAU_construction_Quarter_MRS_rising_HDS_Quarter').data[
        'value'].sum()
    operational_bau_hrs_quarter = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                    scenario='BAU_construction_Quarter_HRS_rising_HDS_Quarter').data[
        'value'].sum()

    ## Sufficiency scenarios
    operational_sufficiency_mrs_constant = \
        combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                          scenario='S2_construction_Constant_MRS_rising_MDS_Constant').data[
            'value'].sum()
    operational_sufficiency_hrs_constant = \
        combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                          scenario='S2_construction_Constant_HRS_rising_MDS_Constant').data[
            'value'].sum()
    operational_sufficiency_mrs_half = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                         scenario='S2_construction_Half_MRS_rising_MDS_Half').data[
        'value'].sum()
    operational_sufficiency_hrs_half = combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                         scenario='S2_construction_Half_HRS_rising_MDS_Half').data[
        'value'].sum()
    operational_sufficiency_mrs_quarter = \
        combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                          scenario='S2_construction_Quarter_MRS_rising_MDS_Quarter').data['value'].sum()
    operational_sufficiency_hrs_quarter = \
        combined_operational_iamdf.filter(year=year, variable='Residential_dwelling',
                                          scenario='S2_construction_Quarter_HRS_rising_MDS_Quarter').data['value'].sum()

    # Filter embodied scenarios with extreme cases
    ## BAU scenarios
    embodied_bau_mrs_re2020 = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                                             scenario='BAU_construction_BAU_RE2020_upfront_MRS_rising_BAU_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_hrs_re2020 = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                                             scenario='BAU_construction_BAU_RE2020_upfront_HRS_rising_BAU_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_mrs_re2020_extended = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                      scenario='BAU_construction_Optimist_RE2020_upfront_MRS_rising_Optimist_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_hrs_re2020_extended = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',

                                                                      scenario='BAU_construction_Optimist_RE2020_upfront_HRS_rising_Optimist_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()

    embodied_sufficiency_mrs_re2020 = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                     scenario='S2_construction_BAU_RE2020_upfront_MRS_rising_BAU_RE2020_WLC_MDS_Constant_WLC').data[
        'value'].sum()
    embodied_sufficiency_hrs_re2020 = combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                                                     scenario='S2_construction_BAU_RE2020_upfront_HRS_rising_BAU_RE2020_WLC_MDS_Constant_WLC').data[
        'value'].sum()
    embodied_sufficiency_mrs_re2020_extended = \
        combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                       scenario='S2_construction_Optimist_RE2020_upfront_MRS_rising_Optimist_RE2020_WLC_MDS_Constant_WLC').data[
            'value'].sum()
    embodied_sufficiency_hrs_re2020_extended = \
        combined_embodied_iamdf.filter(year=year, variable='Residential_dwelling',
                                       scenario='S2_construction_Optimist_RE2020_upfront_HRS_rising_Optimist_RE2020_WLC_MDS_Constant_WLC').data[
            'value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = ['MRS_RE2020_constant_BAU', 'MRS_RE2020_constant_SUF',
                 'MRS_RE2020+_half_BAU', 'MRS_RE2020+_half_SUF',
                 'MRS_RE2020+_quarter_BAU', 'MRS_RE2020+_quarter_SUF',

                 'HRS_RE2020_constant_BAU', 'HRS_RE2020_constant_SUF',
                 'HRS_RE2020+_half_BAU', 'HRS_RE2020+_half_SUF',
                 'HRS_RE2020+_quarter_BAU', 'HRS_RE2020+_quarter_SUF',
                 ]

    operational_values = [operational_bau_mrs_constant, operational_sufficiency_mrs_constant,
                          operational_bau_mrs_half, operational_sufficiency_mrs_half,
                          operational_bau_mrs_quarter, operational_sufficiency_mrs_quarter,

                          operational_bau_hrs_constant, operational_sufficiency_hrs_constant,
                          operational_bau_hrs_half, operational_sufficiency_hrs_half,
                          operational_bau_hrs_quarter, operational_sufficiency_hrs_quarter
                          ]

    embodied_values = [embodied_bau_mrs_re2020, embodied_sufficiency_mrs_re2020,
                       embodied_bau_mrs_re2020_extended, embodied_sufficiency_mrs_re2020_extended,
                       embodied_bau_mrs_re2020_extended, embodied_sufficiency_mrs_re2020_extended,

                       embodied_bau_hrs_re2020, embodied_sufficiency_hrs_re2020,
                       embodied_bau_hrs_re2020_extended, embodied_sufficiency_hrs_re2020_extended,
                       embodied_bau_hrs_re2020_extended, embodied_sufficiency_hrs_re2020_extended
                       ]

    plt.figure(figsize=(20, 15))
    plt.bar(scenarios, operational_values, label='Operational', color='#FFBA08', width=0.3)
    plt.bar(scenarios, embodied_values, label='Embodied', bottom=operational_values, color='#1C3144', width=0.3)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=14, fontweight='bold')
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(f'Whole life GHGE in {year}', fontsize=20, fontweight='bold')
    plt.legend()
    plt.show()


def wlc_graph_sensibility_cum(combined_operational_iamdf, combined_embodied_iamdf):
    # Filter operational scenarios with extreme cases
    ## BAU scenarios
    operational_bau_mrs_constant = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                     scenario='BAU_construction_Constant_MRS_rising_HDS_Constant').data[
        'value'].sum()
    operational_bau_hrs_constant = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                     scenario='BAU_construction_Constant_HRS_rising_HDS_Constant').data[
        'value'].sum()
    operational_bau_mrs_half = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                 scenario='BAU_construction_Half_MRS_rising_HDS_Half').data[
        'value'].sum()
    operational_bau_hrs_half = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                 scenario='BAU_construction_Half_HRS_rising_HDS_Half').data[
        'value'].sum()
    operational_bau_mrs_quarter = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                    scenario='BAU_construction_Quarter_MRS_rising_HDS_Quarter').data[
        'value'].sum()
    operational_bau_hrs_quarter = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                    scenario='BAU_construction_Quarter_HRS_rising_HDS_Quarter').data[
        'value'].sum()

    ## Sufficiency scenarios
    operational_sufficiency_mrs_constant = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                             scenario='S2_construction_Constant_MRS_rising_MDS_Constant').data[
        'value'].sum()
    operational_sufficiency_hrs_constant = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                             scenario='S2_construction_Constant_HRS_rising_MDS_Constant').data[
        'value'].sum()
    operational_sufficiency_mrs_half = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                         scenario='S2_construction_Half_MRS_rising_MDS_Half').data[
        'value'].sum()
    operational_sufficiency_hrs_half = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                         scenario='S2_construction_Half_HRS_rising_MDS_Half').data[
        'value'].sum()
    operational_sufficiency_mrs_quarter = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                            scenario='S2_construction_Quarter_MRS_rising_MDS_Quarter').data[
        'value'].sum()
    operational_sufficiency_hrs_quarter = combined_operational_iamdf.filter(variable='Residential_dwelling',
                                                                            scenario='S2_construction_Quarter_HRS_rising_MDS_Quarter').data[
        'value'].sum()

    # Filter embodied scenarios with extreme cases
    ## BAU scenarios
    embodied_bau_mrs_re2020 = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                             scenario='BAU_construction_BAU_RE2020_upfront_MRS_rising_BAU_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_hrs_re2020 = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                             scenario='BAU_construction_BAU_RE2020_upfront_HRS_rising_BAU_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_mrs_re2020_extended = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                                      scenario='BAU_construction_Optimist_RE2020_upfront_MRS_rising_Optimist_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()
    embodied_bau_hrs_re2020_extended = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                                      scenario='BAU_construction_Optimist_RE2020_upfront_HRS_rising_Optimist_RE2020_WLC_HDS_Constant_WLC').data[
        'value'].sum()

    ## Sufficiency scenarios
    embodied_sufficiency_mrs_re2020 = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                                     scenario='S2_construction_BAU_RE2020_upfront_MRS_rising_BAU_RE2020_WLC_MDS_Constant_WLC').data[
        'value'].sum()
    embodied_sufficiency_hrs_re2020 = combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                                                     scenario='S2_construction_BAU_RE2020_upfront_HRS_rising_BAU_RE2020_WLC_MDS_Constant_WLC').data[
        'value'].sum()
    embodied_sufficiency_mrs_re2020_extended = \
        combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                       scenario='S2_construction_Optimist_RE2020_upfront_MRS_rising_Optimist_RE2020_WLC_MDS_Constant_WLC').data[
            'value'].sum()
    embodied_sufficiency_hrs_re2020_extended = \
        combined_embodied_iamdf.filter(variable='Residential_dwelling',
                                       scenario='S2_construction_Optimist_RE2020_upfront_HRS_rising_Optimist_RE2020_WLC_MDS_Constant_WLC').data[
            'value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = ['MRS_RE2020_constant_BAU', 'MRS_RE2020_constant_SUF',
                 'MRS_RE2020+_half_BAU', 'MRS_RE2020+_half_SUF',
                 'MRS_RE2020+_quarter_BAU', 'MRS_RE2020+_quarter_SUF',

                 'HRS_RE2020_constant_BAU', 'HRS_RE2020_constant_SUF',
                 'HRS_RE2020+_half_BAU', 'HRS_RE2020+_half_SUF',
                 'HRS_RE2020+_quarter_BAU', 'HRS_RE2020+_quarter_SUF',
                 ]

    operational_values = [operational_bau_mrs_constant, operational_sufficiency_mrs_constant,
                          operational_bau_mrs_half, operational_sufficiency_mrs_half,
                          operational_bau_mrs_quarter, operational_sufficiency_mrs_quarter,

                          operational_bau_hrs_constant, operational_sufficiency_hrs_constant,
                          operational_bau_hrs_half, operational_sufficiency_hrs_half,
                          operational_bau_hrs_quarter, operational_sufficiency_hrs_quarter
                          ]

    embodied_values = [embodied_bau_mrs_re2020, embodied_sufficiency_mrs_re2020,
                       embodied_bau_mrs_re2020_extended, embodied_sufficiency_mrs_re2020_extended,
                       embodied_bau_mrs_re2020_extended, embodied_sufficiency_mrs_re2020_extended,

                       embodied_bau_hrs_re2020, embodied_sufficiency_hrs_re2020,
                       embodied_bau_hrs_re2020_extended, embodied_sufficiency_hrs_re2020_extended,
                       embodied_bau_hrs_re2020_extended, embodied_sufficiency_hrs_re2020_extended
                       ]

    plt.figure(figsize=(20, 15))
    plt.bar(scenarios, operational_values, label='Operational', color='#FFBA08', width=0.3)
    plt.bar(scenarios, embodied_values, label='Embodied', bottom=operational_values, color='#1C3144', width=0.3)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=14, fontweight='bold')
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title('Cumulated GHGE', fontsize=20, fontweight='bold')

    # Add horizontal lines at specific y-values
    # plt.axhline(y=17, color='black', linestyle='--', label='SNBC + SDS')

    plt.ylim(0, 2000)
    plt.legend()
    plt.show()



