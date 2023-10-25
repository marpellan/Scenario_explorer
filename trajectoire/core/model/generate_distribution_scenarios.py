import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === FUNCTIONS ===

# Distribution function to generate different scenarios
def generate_distribution(total_renovations, num_years, distribution_type=None, targets=None, mu=None, sigma=None):
    if distribution_type == 'constant':
        return np.full(num_years, total_renovations / num_years)
    elif distribution_type == 'exponential':
        return total_renovations * np.exp(-np.linspace(0, 2, num_years))
    elif distribution_type == 'slow_exponential':
        x = np.linspace(-5, 5, num_years)
        sigmoid = 1 / (1 + np.exp(-x))
        return total_renovations * sigmoid / sigmoid.sum()
    elif distribution_type == 'cyclic':
        x = np.linspace(0, 2 * np.pi, num_years)
        cyclic_values = total_renovations * (np.sin(x) + 1) / 2

        for idx, target in targets.items():
            start_idx = int(idx * num_years / 30)
            end_idx = int((idx + target[1]) * num_years / 30)
            cyclic_values[start_idx:end_idx] = target[0] * total_renovations

        return cyclic_values / cyclic_values.sum()
    elif distribution_type == 'normal':
        if mu is None or sigma is None:
            raise ValueError("For the normal distribution, 'mu' and 'sigma' must be provided.")

        x = np.linspace(0, num_years, num_years)
        normal_values = total_renovations * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return normal_values / normal_values.sum()
    else:
        raise ValueError(
            "Invalid distribution_type. Supported types are 'constant', 'exponential', 'slow_exponential', 'cyclic', and 'normal'.")


def plot_distributions(*dataframes, title_size=14, legend_size=12):
    num_subplots = len(dataframes)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(5, 8))

    for idx, (df, title) in enumerate(dataframes):
        axs[idx].plot(df['Year'], df['Renovations'], label=title)
        axs[idx].set_xlabel('Year')
        axs[idx].set_ylabel('Renovations/year')
        axs[idx].set_title(title, fontsize=title_size)
        axs[idx].legend(fontsize=legend_size)

    plt.tight_layout()
    plt.show()
