import numpy as np
import matplotlib.pyplot as plt

def plot_real_vs_synthetic_sequences(processed_data, synth_data, col_names, num_figures_per_col=3):
    """
    Plots real and synthetic sequences for comparison.

    :param processed_data: Real data as a numpy array.
    :param synth_data: Synthetic data as a numpy array.
    :param col_names: List of column names.
    :param num_figures_per_col: Number of figures per column.
    """
    num = len(col_names)  # Number of columns to plot
    total_subplots = num_figures_per_col * num

    # Create subplots
    fig, axes = plt.subplots(nrows=num_figures_per_col, ncols=num, figsize=(20, 4 * num_figures_per_col))
    axes = axes.flatten() if num_figures_per_col > 1 else [axes]

    # Randomly select indices for sequences to plot
    indices_to_plot = np.random.choice(len(processed_data), total_subplots // num, replace=False)

    # Plot the sequences
    for i, idx in enumerate(indices_to_plot):
        for j, col_name in enumerate(col_names):
            subplot_idx = i * num + j
            real_sequence = processed_data[idx][:, j]
            synthetic_sequence = synth_data[idx][:, j]

            axes[subplot_idx].plot(real_sequence, label='Real Data', color='blue')
            axes[subplot_idx].plot(synthetic_sequence, label='Synthetic Data', color='red', linestyle='--')
            axes[subplot_idx].set_title(f'Variable {col_name}, Sequence {idx}')
            axes[subplot_idx].legend()
            axes[subplot_idx].set_ylim(0, 1)  # Adjust y-axis limits if needed

    # Set common labels and titles
    fig.text(0.5, 0.04, 'Time Step', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Normalized Value', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    return fig