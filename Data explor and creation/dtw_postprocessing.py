# Databricks notebook source
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# COMMAND ----------

# Define the path one level up
parent_directory = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/')
# Add this path to the sys.path list
sys.path.append(f"{parent_directory}")

from utils.Data_splitting import create_split_indices, split_data_with_indices
from utils.Prediction_models import CNN_GRU_regression
from utils.dtw_postprocessing import adjust_all_sequences

from performance_metrics.discriminative_score import calculate_discriminative_scores
from performance_metrics.evaluation_measures import calculate_metrics
from performance_metrics.pearson_corr import calculate_pearson_correlations

from utils.Data_splitting import create_split_indices, split_data_with_indices


# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

#define path to results
num_rows = 0
seq_length = 168

result_file_path = os.path.join(parent_directory, f'results/results_classic_{seq_length}')

filename = f'synthetic_data_flattened_{num_rows}.csv'
full_synth_file_path = os.path.join(result_file_path, filename)

synthetic_flat = pd.read_csv(full_synth_file_path, index_col=None, header=None)

# COMMAND ----------

original_shape = (18602, 168, 1)
reconstructed_synth = np.array(synthetic_flat).reshape(original_shape)
reconstructed_synth.shape

# COMMAND ----------

# import pandas as pd
# import numpy as np

# from mltable import from_delta_lake
# from PIL import Image

# url = "abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/mw08003_complete_extended"

# tl = from_delta_lake(url)

# df = tl.to_pandas_dataframe()
# df.rename(columns = {'Zeitpunkt_x': 'Zeitpunkt'}, inplace=True)
# df.sort_values('Zeitpunkt', inplace= True)
# df['Wert'] = df['Wert'].astype(float)

# df.reset_index(drop=True, inplace=True)

# # Filter the rows where the year in 'Zeitpunkt' 
# df_ = df[df['Zeitpunkt'].dt.year >= 2022]

# attributes = ['Wert']
# feature_n = len(attributes)
# seq_length = 24*7
# time_col = 'Zeitpunkt'

# ### Initialize ###
# # Initialize the Scaling class
# scaler = Scaling(value_cols=attributes)

# # Initialize sequencer
# sequencer = RollingWindow(seq_number=seq_length, time_col=time_col, value_cols=attributes)

# # Initialize preprossing
# preprocessor = Preprocessor(data=df_, normalizer=scaler, sequencer=sequencer)

# processed_data = preprocessor.preprocess()


# COMMAND ----------

# filename = 'wert_flattened.csv'
# # Flattening the array
# flattened_array = processed_data.flatten()

# # Save to CSV after flattening
# np.savetxt(os.path.join(parent_directory, filename), flattened_array, delimiter=',')

# COMMAND ----------

filename = 'Data/wert_flattened.csv'
full_processed_file_path = os.path.join(parent_directory, filename)

processed_flat = pd.read_csv(full_processed_file_path, index_col=None, header=None)
print(processed_flat.shape)
reconstructed_processed = np.array(processed_flat).reshape(original_shape)
reconstructed_processed.shape

# COMMAND ----------

synthetic_dtw_data = adjust_all_sequences(reconstructed_processed, reconstructed_synth)

# COMMAND ----------

# Flattening the array
flattened_dtw = synthetic_dtw_data.flatten()

# Save to CSV after flattening
np.savetxt(os.path.join(result_file_path, f'dtw_flattened_{num_rows}.csv'), flattened_dtw, delimiter=',')

# COMMAND ----------

# Randomly select indices for sequences to plot
num_figures_per_col = 3

indices_to_plot = np.random.choice(len(reconstructed_processed), (num_figures_per_col*2) // 2, replace=False)

# COMMAND ----------

indices_to_plot

# COMMAND ----------

def plot_real_vs_synthetic_sequences(processed_data, synth_data, col_names, random_indices, num_figures_per_col=3):
    """
    Plots real and synthetic sequences for comparison.

    :param processed_data: Real data as a numpy array.
    :param synth_data: Synthetic data as a numpy array.
    :param col_names: List of column names.
    :param random_indices: Numpy array of random indices for sequences to plot.
    :param num_figures_per_col: Number of figures per column.
    :return: Matplotlib figure object.
    """
    num = len(col_names)  # Number of columns to plot
    total_subplots = num_figures_per_col * num

    # Create subplots
    fig, axes = plt.subplots(nrows=num_figures_per_col, ncols=num, figsize=(20, 4 * num_figures_per_col))
    axes = axes.flatten() if num_figures_per_col > 1 else [axes]

    # Plot the sequences
    for i, idx in enumerate(random_indices):
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


# COMMAND ----------

def pca_tsne_real_vs_synth(processed_data, synth_data, sequence_length, idx, sample_size=250):
    real_sample = np.asarray(processed_data)[idx]
    synthetic_sample = np.asarray(synth_data)[idx]

    # Reshape for 2D dimensionality reduction
    processed_data_reduced = real_sample.reshape(-1, sequence_length)
    synth_data_reduced = synthetic_sample.reshape(-1, sequence_length)

    # PCA and t-SNE
    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300)

    pca.fit(processed_data_reduced)
    pca_real = pd.DataFrame(pca.transform(processed_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

    data_reduced = np.concatenate((processed_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

    # Plotting
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # PCA scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('PCA results', fontsize=20, color='red', pad=10)
    ax.scatter(pca_real.iloc[:, 0], pca_real.iloc[:, 1], c='black', alpha=0.2, label='Original')
    ax.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1], c='red', alpha=0.2, label='Synthetic')
    ax.legend()

    # TSNE scatter plot
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('TSNE results', fontsize=20, color='red', pad=10)
    ax2.scatter(tsne_results.iloc[:sample_size, 0], tsne_results.iloc[:sample_size, 1], c='black', alpha=0.2, label='Original')
    ax2.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1], c='red', alpha=0.2, label='Synthetic')
    ax2.legend()

    fig.suptitle('Validating synthetic vs real data diversity and distributions', fontsize=16, color='grey')

    plt.show()

    return fig

# COMMAND ----------

attributes = ['Wert']
fig = plot_real_vs_synthetic_sequences(reconstructed_processed, reconstructed_synth, attributes, indices_to_plot)


# COMMAND ----------

fig.savefig(f'{parent_directory}/dtw1.png')

# COMMAND ----------

fig2 = plot_real_vs_synthetic_sequences(synthetic_dtw_data, reconstructed_synth, attributes, indices_to_plot)

# COMMAND ----------

fig2.savefig(f'{parent_directory}/dtw2.png')

# COMMAND ----------

# Random sampling
sample_size = 250
index = np.random.permutation(len(reconstructed_processed))[:sample_size]

# COMMAND ----------

pca_fig = pca_tsne_real_vs_synth(reconstructed_processed, reconstructed_synth, seq_length, index)

# COMMAND ----------

pca_fig2 = pca_tsne_real_vs_synth(reconstructed_processed, synthetic_dtw_data, seq_length, index)

# COMMAND ----------

# Calculate Pearson correlations
pearson_corrs_synthetic = calculate_pearson_correlations(
    reconstructed_processed, reconstructed_synth
)

print(f'The pearson correlation for synthetic data {pearson_corrs_synthetic}')

# Calculate Pearson correlations
pearson_corrs_synthetic = calculate_pearson_correlations(
    synthetic_dtw_data, reconstructed_synth
)

print(f'The pearson correlation for synthetic dtw data {pearson_corrs_synthetic}')

# COMMAND ----------

target_col_indices = [0]

discriminative_scores = calculate_discriminative_scores(reconstructed_processed, reconstructed_synth, target_col_indices)
print(discriminative_scores)

discriminative_scores = calculate_discriminative_scores(synthetic_dtw_data, reconstructed_synth, target_col_indices)
print(discriminative_scores)

# COMMAND ----------

# Create split indices based on the total number of sequences
total_sequences = reconstructed_processed.shape[0]
train_indices, validation_indices, test_indices = create_split_indices(
    total_samples=total_sequences,
    test_size=0.10,
    validation_size=0.15
)

# Split the data using the indices
X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = split_data_with_indices(
    processed_data=reconstructed_processed,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = split_data_with_indices(
    processed_data=reconstructed_synth,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

X_dtw_train, X_dtw_validation, X_dtw_test, y_dtw_train, y_dtw_validation, y_dtw_test = split_data_with_indices(
    processed_data=synthetic_dtw_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Initialize the model
timesteps = seq_length - 1
units = len(attributes)
train_epoch = 1000

# Initialize real model
real_model = CNN_GRU_regression(timesteps, units, len(target_col_indices))

# Train the model on real

history = real_model.fit(
    X_real_train, y_real_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# Initialize synthetic model
synth_model = CNN_GRU_regression(timesteps, units, len(target_col_indices))

# Train the model
synth_history = synth_model.fit(
    X_synth_train, y_synth_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# Initialize dtw synthetic model
dtw_model = CNN_GRU_regression(timesteps, units, len(target_col_indices))

# Train the model
dtw_history = dtw_model.fit(
    X_dtw_train, y_dtw_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# COMMAND ----------

print('for synth')
r_pred = real_model.predict(X_real_test)
s_pred = synth_model.predict(X_real_test)
dtw_pred = dtw_model.predict(X_real_test)

# COMMAND ----------

print('compared synthetic only wert')
r2_scores_and_mae = calculate_metrics(y_real_test, r_pred, s_pred)
print(r2_scores_and_mae)

# COMMAND ----------

r2_scores_and_mae.display()

# COMMAND ----------

print('compared synthetic only wert')
r2_scores_and_mae = calculate_metrics(y_real_test, r_pred, dtw_pred)
print(r2_scores_and_mae)
