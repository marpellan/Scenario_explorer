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

nrj_types = ['Electricity (elec)', 'Wood (bois)', 'Oil (fioul)', 'Natural Gas (gaz)', 'LPG (gpl)', 'Coal (charbon)', 'District Heating (rcu)']
#colors_nrj = ['#0077BE', '#8B4513', '#343434', '#40E0D0', '#800080', '#505050', '#FF4500']

nrj_mapping ={
    'bois': 'biomass',
    'elec': 'electricity',
    'fioul': 'oil products',
    'gaz': 'natural gas',
    'gpl': 'lpg',
    'charbon': 'coal',
    'rcu': 'district heating'
}


energy_colors = {
    'biomass': '#90be6d',
    'electricity': '#FFBA08',
    'oil products': '#550000',
    'natural gas': '#d8cba7',
    'district heating': '#264653',
    'lpg': '#555555',
    'coal': '#997b66'
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


def plot_operational_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None,save_path=None):
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

    plt.legend(loc='upper right', fontsize=14)

    plt.title(title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_cumulative_operational_stock_dpe(iamdf, scenario_pattern=None, variable=None, title=None, figsize=None, save_path=None):
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

    plt.legend(loc='upper left', fontsize=14)

    plt.title(title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_embodied_all_stock(df, title=None, save_path=None):
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
    plt.legend(loc='upper right', fontsize=14)
    plt.title(title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_cumulative_all_stock_embodied(iamdf, title=None, save_path=None):
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
    plt.legend(loc='upper left', fontsize=14)
    plt.title(title, fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlabel("Year", fontsize=16, fontweight="bold")
    plt.ylabel("MtCO2eq", fontsize=16, fontweight="bold")

    plt.ylim(0, 400)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_operational_all_stock_dpe(df, scenario_pattern, variable, title, figsize, save_path=None):
    if figsize is None:
        figsize = (20, 10)

    # Filter based on scenario and variable
    filtered_df = df.filter(scenario=scenario_pattern, variable=variable)
    filtered_df = filtered_df.timeseries().reset_index()

    # Extract year columns
    year_columns = [col for col in filtered_df.columns if isinstance(col, int)]

    # Define custom colors for each DPE value
    custom_colors = {
        'A': '#3F3F3D', 'B': '#5B755D', 'C': '#8CA188', 'D': '#E1BA75',
        'E': '#EE892F', 'F': '#E76020', 'G': '#ae251b', 'new': '#cdd7d6'
    }

    # Sort DPE values according to the custom color order
    sorted_dpe = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'new']

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, fontweight="bold")
    #ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel("MtCO2eq", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 60)  # Set the y-axis range

    # Initialize bottom array for stacked bars
    bottom = [0] * len(year_columns)

    # Stack bars for each DPE
    for dpe in sorted_dpe:
        dpe_values = filtered_df[filtered_df['dpe'] == dpe][year_columns].sum()
        ax.bar(year_columns, dpe_values, bottom=bottom, label=dpe, color=custom_colors[dpe])
        # Update bottom values for the next stack
        bottom = [sum(x) for x in zip(bottom, dpe_values)]

    ax.legend(title="DPE", loc='upper right', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_operational_all_stock_nrj(df, scenario_pattern, variable, title, figsize=None, save_path=None):
    if figsize is None:
        figsize = (20, 10)

    # Filter based on scenario and variable
    filtered_df = df.filter(scenario=scenario_pattern, variable=variable)
    filtered_df = filtered_df.timeseries().reset_index()

    # Translate energy names using nrj_mapping
    nrj_mapping = {
        'bois': 'biomass',
        'elec': 'electricity',
        'fioul': 'oil products',
        'gaz': 'natural gas',
        'gpl': 'lpg',
        'charbon': 'coal',
        'rcu': 'district heating'
    }
    filtered_df['ef_nrj'] = filtered_df['ef_nrj'].map(nrj_mapping)

    # Extract year columns
    year_columns = [col for col in filtered_df.columns if isinstance(col, int)]

    # Define custom colors for each EF_NRJ value
    energy_colors = {
        'biomass': '#90be6d',
        'electricity': '#FFBA08',
        'oil products': '#550000',
        'natural gas': '#d8cba7',
        'district heating': '#264653',
        'lpg': '#555555',
        'coal': '#997b66'
    }

    # Ensure all EF_NRJ values are in the energy_colors dictionary
    unique_ef_nrj = filtered_df['ef_nrj'].unique()
    ef_nrj_colors = {ef_nrj: energy_colors.get(ef_nrj, "gray") for ef_nrj in unique_ef_nrj}  # Default to gray if not found

    # Sort EF_NRJ values for consistent order
    sorted_ef_nrj = sorted(unique_ef_nrj)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, fontweight="bold")
    #ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel("MtCO2eq", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 60)  # Set the y-axis range

    # Initialize bottom array for stacked bars
    bottom = [0] * len(year_columns)

    # Stack bars for each EF_NRJ
    for ef_nrj in sorted_ef_nrj:
        ef_nrj_values = filtered_df[filtered_df['ef_nrj'] == ef_nrj][year_columns].sum()
        ax.bar(year_columns, ef_nrj_values, bottom=bottom, label=ef_nrj, color=ef_nrj_colors[ef_nrj])
        # Update bottom values for the next stack
        bottom = [sum(x) for x in zip(bottom, ef_nrj_values)]

    ax.legend(title="EF_NRJ", loc='upper right', fontsize=14)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_stock_nrj_twh(df, scenario_pattern, variable, title, figsize=None, save_path=None):
    if figsize is None:
        figsize = (20, 10)

    # Filter based on scenario and variable
    filtered_df = df.filter(scenario=scenario_pattern, variable=variable)
    filtered_df = filtered_df.timeseries().reset_index()

    # Translate energy names using nrj_mapping
    nrj_mapping = {
        'bois': 'biomass',
        'elec': 'electricity',
        'fioul': 'oil products',
        'gaz': 'natural gas',
        'gpl': 'lpg',
        'charbon': 'coal',
        'rcu': 'district heating'
    }
    # Convert ef_nrj values to strings and then apply the mapping
    filtered_df['ef_nrj'] = filtered_df['ef_nrj'].astype(str).map(nrj_mapping)

    # Exclude 'calibrated' from the ef_nrj column (assuming 'calibrated' is the mapped value)
    filtered_df = filtered_df[filtered_df['ef_nrj'] != 'calibrated']

    # Extract year columns
    year_columns = [col for col in filtered_df.columns if isinstance(col, int)]


    # Define custom colors for each EF_NRJ value
    energy_colors = {
        'biomass': '#90be6d',
        'electricity': '#FFBA08',
        'oil products': '#550000',
        'natural gas': '#d8cba7',
        'district heating': '#264653',
        'lpg': '#555555',
        'coal': '#997b66'
    }

    # Ensure all EF_NRJ values are in the energy_colors dictionary
    unique_ef_nrj = filtered_df['ef_nrj'].unique()
    ef_nrj_colors = {ef_nrj: energy_colors.get(ef_nrj, "gray") for ef_nrj in unique_ef_nrj}  # Default to gray if not found

    # Sort EF_NRJ values for consistent order
    #sorted_ef_nrj = sorted(unique_ef_nrj)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, fontweight="bold")
    #ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel("TWh", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 600)  # Set the y-axis range

    # Initialize bottom array for stacked bars
    bottom = [0] * len(year_columns)

    # Stack bars for each EF_NRJ
    for ef_nrj in filtered_df['ef_nrj'].unique(): #sorted_ef_nrj:
        ef_nrj_values = filtered_df[filtered_df['ef_nrj'] == ef_nrj][year_columns].sum() / 10e8
        ax.bar(year_columns, ef_nrj_values, bottom=bottom, label=ef_nrj, color=ef_nrj_colors[ef_nrj])
        # Update bottom values for the next stack
        bottom = [sum(x) for x in zip(bottom, ef_nrj_values)]

    ax.legend(title="EF_NRJ", loc='upper right', fontsize=14)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    #plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_cumulative_operational_all_stock_dpe(df, scenario_pattern, variable, title, figsize, save_path=None):
    if figsize is None:
        figsize = (20, 10)

    # Filter based on scenario and variable
    filtered_df = df.filter(scenario=scenario_pattern, variable=variable)
    filtered_df = filtered_df.timeseries().reset_index()

    # Extract year columns
    year_columns = [col for col in filtered_df.columns if isinstance(col, int)]

    # Define custom colors for each DPE value
    custom_colors = {
        'A': '#3F3F3D', 'B': '#5B755D', 'C': '#8CA188', 'D': '#E1BA75',
        'E': '#EE892F', 'F': '#E76020', 'G': '#ae251b', 'new': '#cdd7d6'
    }

    # Sort DPE values according to the custom color order
    sorted_dpe = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'new']

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, fontweight="bold")
    #ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative MtCO2eq", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1400)  # Set the y-axis range

    # Initialize bottom array for stacked bars
    bottom = [0] * len(year_columns)

    # Stack bars for each DPE, calculating cumulative sum each year
    for dpe in sorted_dpe:
        dpe_values = filtered_df[filtered_df['dpe'] == dpe][year_columns].sum().cumsum()
        ax.bar(year_columns, dpe_values, bottom=bottom, label=dpe, color=custom_colors[dpe])
        # Update bottom values for the next stack
        bottom = [sum(x) for x in zip(bottom, dpe_values)]

    ax.legend(title="DPE", loc='upper left', fontsize=14)
    #plt.xticks(rotation=45)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_cumulative_operational_all_stock_nrj(df, scenario_pattern, variable, title, figsize=None, save_path=None):
    if figsize is None:
        figsize = (20, 10)

    # Filter based on scenario and variable
    filtered_df = df.filter(scenario=scenario_pattern, variable=variable)
    filtered_df = filtered_df.timeseries().reset_index()

    # Convert ef_nrj values to strings and then apply the mapping
    nrj_mapping = {
        'bois': 'biomass',
        'elec': 'electricity',
        'fioul': 'oil products',
        'gaz': 'natural gas',
        'gpl': 'lpg',
        'charbon': 'coal',
        'rcu': 'district heating'
    }
    filtered_df['ef_nrj'] = filtered_df['ef_nrj'].astype(str).map(nrj_mapping)

    # Extract year columns
    year_columns = [col for col in filtered_df.columns if isinstance(col, int)]

    # Define custom colors for each EF_NRJ value
    energy_colors = {
        'biomass': '#90be6d',
        'electricity': '#FFBA08',
        'oil products': '#550000',
        'natural gas': '#d8cba7',
        'district heating': '#264653',
        'lpg': '#555555',
        'coal': '#997b66'
    }

    # Ensure all EF_NRJ values are in the energy_colors dictionary
    unique_ef_nrj = filtered_df['ef_nrj'].unique()
    ef_nrj_colors = {ef_nrj: energy_colors.get(ef_nrj, "gray") for ef_nrj in unique_ef_nrj}  # Default to gray if not found

    # Sort EF_NRJ values for consistent order
    sorted_ef_nrj = sorted(unique_ef_nrj)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, fontweight="bold")
    #ax.set_xlabel("Year", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative MtCO2eq", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1400)  # Set the y-axis range

    # Initialize bottom array for stacked bars
    bottom = [0] * len(year_columns)

    # Stack bars for each EF_NRJ, calculating cumulative sum each year
    for ef_nrj in sorted_ef_nrj:
        ef_nrj_values = filtered_df[filtered_df['ef_nrj'] == ef_nrj][year_columns].sum().cumsum()
        ax.bar(year_columns, ef_nrj_values, bottom=bottom, label=ef_nrj, color=ef_nrj_colors[ef_nrj])
        # Update bottom values for the next stack
        bottom = [sum(x) for x in zip(bottom, ef_nrj_values)]

    ax.legend(title="EF_NRJ", loc='upper left', fontsize=14)
    #plt.xticks(rotation=45)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def wlc_graph_compare_scenarios(operational_bau, operational_optimistic,
                                embodied_bau, embodied_optimistic,
                                scenario_name_1 = None, scenario_name_2 = None,
                                variable=None,
                                year=None,
                                save_path=None):

    # Calculate the operational and embodied values for each scenario
    scope_1_bau_value = operational_bau.filter(year=year, variable=variable, ef_nrj=['bois', 'charbon', 'fioul', 'gpl', 'gaz']).data['value'].sum()
    scope_2_bau_value = operational_bau.filter(year=year, variable=variable, ef_nrj=['elec', 'rcu']).data['value'].sum()
    scope_3_bau_value = embodied_bau.filter(year=year, variable=variable).data['value'].sum()

    scope_1_optimistic_value = operational_optimistic.filter(year=year, variable=variable, ef_nrj=['bois', 'charbon', 'fioul', 'gpl', 'gaz']).data['value'].sum()
    scope_2_optimistic_value = operational_optimistic.filter(year=year, variable=variable, ef_nrj=['elec', 'rcu']).data['value'].sum()
    scope_3_optimistic_value = embodied_optimistic.filter(year=year, variable=variable).data['value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = [scenario_name_1, scenario_name_2]
    scope_1_values = [scope_1_bau_value, scope_1_optimistic_value]
    scope_2_values = [scope_2_bau_value, scope_2_optimistic_value]
    scope_3_values = [scope_3_bau_value, scope_3_optimistic_value]

    s3_values = [s1 + s2 for s1, s2 in zip(scope_1_values, scope_2_values)]


    plt.figure(figsize=(10, 8))
    plt.bar(scenarios, scope_1_values, label='Direct operational GHGE', color='#D00000')
    plt.bar(scenarios, scope_2_values, label='Indirect operational GHGE', color='#FFBA08', bottom=scope_1_values)
    plt.bar(scenarios, scope_3_values, label='Embodied GHGE', bottom=s3_values, color='#1C3144')
    #plt.xticks(rotation=45)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(f'Whole life GHGE in {year}', fontsize=20, fontweight='bold')

    # Add horizontal lines at specific y-values
    #plt.axhline(y=27.7, color='black', linestyle='--', label='SNBC + APS')
    #plt.axhline(y=8.2, color='black', linestyle='--', label='SNBC + NZS')

    plt.legend(fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def wlc_graph_compare_scenarios_cum(operational_bau, operational_optimistic,
                                embodied_bau, embodied_optimistic,
                                scenario_name_1 = None, scenario_name_2 = None,
                                variable=None,
                                save_path=None):

    # Calculate the operational and embodied values for each scenario
    scope_1_bau_value = operational_bau.filter(variable=variable, ef_nrj=['bois', 'charbon', 'fioul', 'gpl', 'gaz']).data['value'].sum()
    scope_2_bau_value = operational_bau.filter(variable=variable, ef_nrj=['elec', 'rcu']).data['value'].sum()
    scope_3_bau_value = embodied_bau.filter(variable=variable).data['value'].sum()

    scope_1_optimistic_value = operational_optimistic.filter(variable=variable, ef_nrj=['bois', 'charbon', 'fioul', 'gpl']).data['value'].sum()
    scope_2_optimistic_value = operational_optimistic.filter(variable=variable, ef_nrj=['elec', 'rcu']).data['value'].sum()
    scope_3_optimistic_value = embodied_optimistic.filter(variable=variable).data['value'].sum()

    # Create a stacked bar chart with custom colors
    scenarios = [scenario_name_1, scenario_name_2]
    scope_1_values = [scope_1_bau_value, scope_1_optimistic_value]
    scope_2_values = [scope_2_bau_value, scope_2_optimistic_value]
    scope_3_values = [scope_3_bau_value, scope_3_optimistic_value]

    s3_values = [s1 + s2 for s1, s2 in zip(scope_1_values, scope_2_values)]


    plt.figure(figsize=(10, 8))
    plt.bar(scenarios, scope_1_values, label='Direct operational GHGE', color='#D00000')
    plt.bar(scenarios, scope_2_values, label='Indirect operational GHGE', color='#FFBA08', bottom=scope_1_values)
    plt.bar(scenarios, scope_3_values, label='Embodied GHGE', bottom=s3_values, color='#1C3144')
    #plt.xticks(rotation=45)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    # plt.xlabel('Scenario')
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(f'Cumulative Whole life GHGE', fontsize=20, fontweight='bold')

    # Add horizontal lines at specific y-values
    #plt.axhline(y=27.7, color='black', linestyle='--', label='SNBC + APS')
    #plt.axhline(y=8.2, color='black', linestyle='--', label='SNBC + NZS')

    plt.legend(fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def wlc_graph_sensibility(combined_operational_iamdf, combined_embodied_iamdf, year=None, save_path=None):
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
    plt.legend(fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def wlc_graph_sensibility_cum(combined_operational_iamdf, combined_embodied_iamdf, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_range_operational_ghge(iamdf, variable=None, scenario=None, title=None, save_path=None):
    # Filter based on variable and transform into df
    filtered_df = iamdf.filter(variable=variable, scenario=scenario)
    df = filtered_df.timeseries().reset_index()

    plt.figure(figsize=(10, 8))

    # Extract year columns for the x-axis
    year_columns = [year for year in range(2021, 2052)]

    # Define colors and styles for keywords
    color_keywords = {'Constant': '#0c1618', 'Half': '#004643', 'SNBC': '#c8d5b9'}
    style_keywords = {'linear': 'solid', 'plateau': 'dashed'}

    # Set for unique labels
    unique_labels = set()

    # Sum emissions across categories for each year
    df_summed = df.groupby(['model', 'scenario', 'region', 'variable', 'unit'])[year_columns].sum().reset_index()

    # Plot each scenario
    for scenario in df_summed['scenario'].unique():
        scenario_data = df_summed[df_summed['scenario'] == scenario].iloc[0]
        color = 'grey'  # Default color
        linestyle = 'solid'  # Default linestyle
        marker = None

        # Assign color and linestyle based on keywords
        label_parts = []
        for keyword, color_value in color_keywords.items():
            if keyword in scenario:
                color = color_value
                label_parts.append(keyword)
                break

        for keyword, line_style in style_keywords.items():
            if keyword in scenario:
                linestyle = line_style
                marker = 'o' if line_style == 'dashed' else None
                label_parts.append(keyword)
                break

        # Construct label
        label = " ".join(label_parts)

        # Plot and add to legend if label is unique
        if label and label not in unique_labels:
            plt.plot(year_columns, scenario_data[year_columns], label=label, color=color, linestyle=linestyle, marker=marker)
            unique_labels.add(label)
        else:
            plt.plot(year_columns, scenario_data[year_columns], color=color, linestyle=linestyle, marker=marker)

    # Plot formatting
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(title if title else "Operational range of values", fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 60)
    plt.tight_layout()  # Adjust plot layout

    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()


def plot_range_embodied_ghge(iamdf, variable=None, scenario=None, title=None, save_path=None):
    # Filter based on variable and transform into df
    filtered_df = iamdf.filter(variable=variable, scenario=scenario)
    df = filtered_df.timeseries().reset_index()

    plt.figure(figsize=(10, 8))

    # Extract year columns for the x-axis
    year_columns = [year for year in range(2020, 2051)]

    # Define colors and styles for keywords
    color_keywords = {'RE2020plus': '#2a9d8f', 'RE2020': '#264653'}
    style_keywords = {'linear': 'solid', 'plateau': 'dashed'}

    # Set for unique labels
    unique_labels = set()

    # Sum emissions across categories for each year
    df_summed = df.groupby(['model', 'scenario', 'region', 'variable', 'unit'])[year_columns].sum().reset_index()

    # Plot each scenario
    for scenario in df_summed['scenario'].unique():
        scenario_data = df_summed[df_summed['scenario'] == scenario].iloc[0]
        color = 'grey'  # Default color
        linestyle = 'solid'  # Default linestyle
        marker = None

        # Assign color and linestyle based on keywords
        label_parts = []
        for keyword, color_value in color_keywords.items():
            if keyword in scenario:
                color = color_value
                label_parts.append(keyword)
                break

        for keyword, line_style in style_keywords.items():
            if keyword in scenario:
                linestyle = line_style
                marker = 'o' if line_style == 'dashed' else None
                label_parts.append(keyword)
                break

        # Construct label
        label = " ".join(label_parts)

        # Plot and add to legend if label is unique
        if label and label not in unique_labels:
            plt.plot(year_columns, scenario_data[year_columns], label=label, color=color, linestyle=linestyle, marker=marker)
            unique_labels.add(label)
        else:
            plt.plot(year_columns, scenario_data[year_columns], color=color, linestyle=linestyle, marker=marker)

    # Plot formatting
    plt.ylabel('MtCO2eq', fontsize=16, fontweight='bold')
    plt.title(title if title else "Embodied ranges of values", fontsize=20, fontweight='bold')
    plt.legend(loc='lower left', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()  # Adjust plot layout

    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()


def plot_stock_dynamics(df, scenario_pattern, save_path=None):
    # Define colors
    colors = {
        'New construction': '#e07a5f',
        'Renovation': '#f3d8c7',
        'Demolition': '#3d405b'
    }

    # Filter the dataframe for the chosen scenario
    filtered_df = df.filter(scenario=scenario_pattern)
    filtered_df = filtered_df.timeseries().reset_index()

    # Extract year columns
    years = [col for col in filtered_df.columns if isinstance(col, int)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for variable in filtered_df['variable'].unique():
        if variable != 'Population':  # Exclude Population from the primary axis
            data = filtered_df[filtered_df['variable'] == variable][years].T
            ax.plot(years, data, label=variable, color=colors.get(variable, 'blue'))

    ax.set_ylabel(df['unit'].iloc[0], fontsize=14, labelpad=10)
    ax.set_title(f'Stock dynamics in {scenario_pattern} scenario', fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1300000)  # Set the y-axis range
    ax.tick_params(axis='x', labelsize=13)

    # Secondary axis for Population
    ax2 = ax.twinx()
    if 'Population' in filtered_df['variable'].unique():
        population_data = filtered_df[filtered_df['variable'] == 'Population'][years].T
        pop_line, = ax2.plot(years, population_data, label='Population', color='black', linestyle='dashed')
        ax2.set_ylabel('Population', fontsize=14, labelpad=10)

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    if 'Population' in filtered_df['variable'].unique():
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + [pop_line], labels + labels2, loc='upper left', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

