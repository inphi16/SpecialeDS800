# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

# %pip install tbats
# from tbats import TBATS

import pyspark as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose



# COMMAND ----------

df = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/mw08003_complete_extended")
df.sort_values('Zeitpunkt_x', inplace= True)
df = df.drop(columns = ['Infopunkt'])
df.rename(columns = {'Zeitpunkt_x': 'Zeitpunkt'}, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # TimeSeries Decomposition

# COMMAND ----------

# def analyze_time_series(df, timestamp_col, value_col, seasonal_periods=None):
#     # Set the timestamp column as the index
#     if timestamp_col in df.columns:
#         df.set_index(timestamp_col, inplace=True)
    
#     # Plot autocorrelation
#     plot_acf(df[value_col])
#     plt.show()
    
#     # Plot autocorrelation with more lags
#     plot_acf(df[value_col], lags=168, alpha=0.05)
#     plt.show()
    
#     # Decompose the time series
#     decomposition = seasonal_decompose(df[value_col], model='additive', period=8760)
#     decomposition.plot()
#     plt.show()

#     # Use default periods if none are provided
#     if seasonal_periods is None:
#         seasonal_periods = [24, 168]

#     # Perform MSTL decomposition
#     mstl = MSTL(df[value_col], periods=seasonal_periods)
#     res = mstl.fit()
    
#     # Plot the seasonal components
#     fig, ax = plt.subplots(nrows=len(seasonal_periods), figsize=[10, 10])

#     for i, period in enumerate(seasonal_periods):
#         seasonality_label = f"seasonal_{period}"
#         res.seasonal[seasonality_label].iloc[:period*3].plot(ax=ax[i])
#         ax[i].set_ylabel(seasonality_label)
#         ax[i].set_title(f"Seasonality for period: {period}")

#     plt.tight_layout()
#     plt.show()

# analyze_time_series(df, 'Zeitpunkt', 'Wert', seasonal_periods=[24, 24*365])

# COMMAND ----------

from statsmodels.tsa.seasonal import STL


def remove_seasonality(df, timestamp_col, value_col, period):
    # Ensure that the timestamp column is of datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set the timestamp column as the index
    df.set_index(timestamp_col, inplace=True)
    
    # Decompose the time series with STL
    stl = STL(df[value_col], period=period, robust=True)
    result = stl.fit()
    
    # Add the deseasonalized and seasonal components to the DataFrame
    df['deseasonalized_wert'] = df[value_col] - result.seasonal
    df['24seasonality'] = result.seasonal
    
    # Reset index to bring the timestamp back as a column if needed
    df.reset_index(inplace=True)
    
    return df

# Call the function with period=24 to remove daily seasonality
df_with_seasonality = remove_seasonality(df, 'Zeitpunkt', 'Wert', 24*7)



# COMMAND ----------

# Filter the rows where the year in 'Zeitpunkt' is 2023
# df_2023 = df_with_seasonality[df_with_seasonality['Zeitpunkt'].dt.year == 2023]
df_2023 = df[df['Zeitpunkt'].dt.year == 2023]

# COMMAND ----------

# MAGIC %run Ines_TimeGAN_func

# COMMAND ----------

import pyspark as ps
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import GRU, Dense

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import backend as K
from tqdm import tqdm
from scipy.spatial.distance import euclidean


# COMMAND ----------

feature_n = 2
seq_len = 24*7
column_names = ['Wert', 'temp'] #, 'day_type_indicator', 'daily_median_temp', 'daily_std_temp', 'hourly_avg_wert', 'weekly_avg_wert'] #, 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']

# Defining model and training parameters
model_args = ModelParameters(batch_size=100,
                             lr=0.001,
                             betas=(0.2, 0.9),
                             latent_dim=20,
                             gp_lambda=2,
                             pac=1,
                             hidden_dim=feature_n)
 
train_args = TrainParameters(epochs=1000, 
                             sequence_length=seq_len,
                             sample_length=8,
                             rounds=1,
                             measurement_cols=column_names,
                             number_sequences = feature_n
                             )

# COMMAND ----------

time_gan = TimeGAN(model_args)

# COMMAND ----------

# # Assuming all arrays are numpy arrays and have compatible shapes
# is_equal = np.isclose(
#     sequences_array_full[:, :, 1],  # Yellow: Normalized 'Wert'
#     (sequences_array_full[:, :, 2] - sequences_array_full[:, :, 3]),  # Orange + Green
#     atol=1e-8  # Tolerance level, can be adjusted based on the precision of your data
# )

# COMMAND ----------

c_names = ['Zeitpunkt', 'Wert', 'temp', 'day_type_indicator', 'daily_median_temp', 'daily_std_temp', 'hourly_avg_wert', 'weekly_avg_wert']
df_2023 = df_2023[c_names]
df_2023


# COMMAND ----------

processed_data, full, seasonality, scalers = time_gan.fit(df_2023, train_args, num_cols = column_names)

# COMMAND ----------

print(full.shape,
processed_data.shape)

# COMMAND ----------

# Number of samples to generate
synthetic_data = np.asarray(time_gan.sample(len(processed_data)))

# COMMAND ----------

# MAGIC %run Visualizations_func

# COMMAND ----------

plot_real_vs_synthetic_sequences(processed_data, synthetic_data, column_names)

# COMMAND ----------

pca_tsne_real_vs_synth(processed_data, synthetic_data, seq_len)

# COMMAND ----------

# Calculate Pearson correlations
pearson_corrs_synthetic = calculate_pearson_correlations(
    processed_data, synthetic_data
)


print(f'The pearson correlation for synthetic data {pearson_corrs_synthetic}')

# COMMAND ----------

def reseasonalize_data(generated_sequences, seasonal_components, seq_len, num_cols):
    # Initialize container for reseasonalized data
    reseasonalized_data = []

    # Add back the seasonality for each sequence
    for i in range(len(generated_sequences)):
        sequence = generated_sequences[i]
        
        # Retrieve the seasonal component for each feature
        seasonal_sequence = np.array([seasonal_components[col][i] for col in num_cols]).T
        
        # Reintroduce the seasonality by adding the seasonal component to the generated data
        reseasonalized_sequence = sequence + seasonal_sequence
        reseasonalized_data.append(reseasonalized_sequence)

    # Convert the list of sequences to a 3D numpy array
    reseasonalized_data = np.array(reseasonalized_data)

    return reseasonalized_data


# COMMAND ----------

from sklearn.metrics import pairwise_distances

def calculate_mmd(X, Y, gamma=None):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two samples, X and Y.
    The kernel can be computed using a Gaussian kernel with bandwidth gamma.
    If gamma is None, it uses the median heuristic.
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    # Compute the pairwise distances in the Gaussian kernel
    XX = pairwise_distances(X.reshape(n, -1), metric='euclidean')
    YY = pairwise_distances(Y.reshape(m, -1), metric='euclidean')
    XY = pairwise_distances(X.reshape(n, -1), Y.reshape(m, -1), metric='euclidean')
    
    if gamma is None:
        # Median heuristic for the bandwidth
        gamma = np.median(np.hstack((XX.flatten(), YY.flatten(), XY.flatten())))
    
    # Compute the kernel matrices
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)
    
    # Compute the MMD
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd

# Compute MMD between the real and synthetic datasets
mmd_value = calculate_mmd(processed_data, synthetic_data)
print(f"MMD: {mmd_value}")


# COMMAND ----------

from scipy.stats import ks_2samp
from scipy.spatial.distance import cityblock

def calculate_ks_test(sample1, sample2):
    # Perform the KS test
    ks_statistic, p_value = ks_2samp(sample1, sample2)
    return ks_statistic, p_value

def calculate_tvd(p, q):
    # The TVD is half the L1-norm (Manhattan distance) between two probability distributions
    tvd = cityblock(p, q) / 2.0
    return tvd

# Assuming your datasets are named 'processed_data' and 'synthetic_data'
# with shape (8737, 24, 2)
# Let's loop over each attribute

num_attributes = processed_data.shape[2]  # In your case, 2

# Store the results
ks_results = []
tvd_results = []

for attribute_index in range(num_attributes):
    # Extracting the attribute across all sequences for both datasets
    real_attribute_data = processed_data[:, :, attribute_index].flatten()
    synthetic_attribute_data = synthetic_data[:, :, attribute_index].flatten()
    
    # KS Test for the current attribute
    ks_statistic, p_value = calculate_ks_test(real_attribute_data, synthetic_attribute_data)
    ks_results.append((ks_statistic, p_value))
    
    # TVD for the current attribute
    # Get the histograms as probability distributions for each attribute
    p = np.histogram(real_attribute_data, bins=50, range=(0, 1), density=True)[0]
    q = np.histogram(synthetic_attribute_data, bins=50, range=(0, 1), density=True)[0]
    
    tvd = calculate_tvd(p, q)
    tvd_results.append(tvd)

# Now, ks_results and tvd_results hold the KS statistics/p-values and TVDs for each attribute
for i, (ks, tvd) in enumerate(zip(ks_results, tvd_results)):
    print(f"Attribute {i}: KS statistic: {ks[0]}, P-value: {ks[1]}, TVD: {tvd}")


# COMMAND ----------

seasonality_processed = reseasonalize_data(processed_data, seasonality, seq_len, column_names)
Seasonality_synth = reseasonalize_data(synthetic_data, seasonality, seq_len, column_names)

# COMMAND ----------

plot_real_vs_synthetic_sequences(seasonality_processed, Seasonality_synth, column_names)

# COMMAND ----------

pca_tsne_real_vs_synth(seasonality_processed, Seasonality_synth, seq_len)

# COMMAND ----------

# Calculate Pearson correlations
pearson_corrs_synthetic = calculate_pearson_correlations(
    seasonality_processed, Seasonality_synth
)


print(f'The pearson correlation for synthetic data {pearson_corrs_synthetic}')

# COMMAND ----------

# Compute MMD between the real and synthetic datasets
mmd_value = calculate_mmd(seasonality_processed, Seasonality_synth)
print(f"MMD: {mmd_value}")

# COMMAND ----------

from scipy.stats import ks_2samp
from scipy.spatial.distance import cityblock

def calculate_ks_test(sample1, sample2):
    # Perform the KS test
    ks_statistic, p_value = ks_2samp(sample1, sample2)
    return ks_statistic, p_value

def calculate_tvd(p, q):
    # The TVD is half the L1-norm (Manhattan distance) between two probability distributions
    tvd = cityblock(p, q) / 2.0
    return tvd

# Assuming your datasets are named 'processed_data' and 'synthetic_data'
# with shape (8737, 24, 2)
# Let's loop over each attribute

num_attributes = processed_data.shape[2]  # In your case, 2

# Store the results
ks_results = []
tvd_results = []

for attribute_index in range(num_attributes):
    # Extracting the attribute across all sequences for both datasets
    real_attribute_data = seasonality_processed[:, :, attribute_index].flatten()
    synthetic_attribute_data = Seasonality_synth[:, :, attribute_index].flatten()
    
    # KS Test for the current attribute
    ks_statistic, p_value = calculate_ks_test(real_attribute_data, synthetic_attribute_data)
    ks_results.append((ks_statistic, p_value))
    
    # TVD for the current attribute
    # Get the histograms as probability distributions for each attribute
    p = np.histogram(real_attribute_data, bins=50, range=(0, 1), density=True)[0]
    q = np.histogram(synthetic_attribute_data, bins=50, range=(0, 1), density=True)[0]
    
    tvd = calculate_tvd(p, q)
    tvd_results.append(tvd)

# Now, ks_results and tvd_results hold the KS statistics/p-values and TVDs for each attribute
for i, (ks, tvd) in enumerate(zip(ks_results, tvd_results)):
    print(f"Attribute {i}: KS statistic: {ks[0]}, P-value: {ks[1]}, TVD: {tvd}")


# COMMAND ----------

# # Assuming all arrays are numpy arrays and have compatible shapes
# is_equal = np.isclose(
#     full_processed_data[:, :, 1],  # Yellow: Normalized 'Wert'
#     (full_processed_data[:, :, 2] - full_processed_data[:, :, 3]),  # Orange + Green
#     atol=1e-8  # Tolerance level, can be adjusted based on the precision of your data
# )

# is_equal
# normalized_seasonality = full_processed_data[:, :, 3]
# df_processed = processed_data.copy()
# df_synthetic = synthetic_data.copy()
# df_processed[:,:,1] = df_processed[:,:,1] + normalized_seasonality
# df_synthetic[:,:,1] =  df_synthetic[:,:,1] + normalized_seasonality
# is_equal = np.isclose(
#     full_processed_data[:, :, 2], df_processed[:,:,1] 
# )
# is_equal

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction

# COMMAND ----------

# MAGIC %run models_func

# COMMAND ----------

def create_split_indices(total_samples, test_size, validation_size, chronological=True):
    """
    Create indices for the training, validation, and testing splits.
    """
    if chronological:
        # If chronological, don't shuffle and split the indices in order
        test_end = int(total_samples * (1 - test_size))
        validation_end = int(test_end * (1 - validation_size))
        
        train_indices = np.arange(0, validation_end)
        validation_indices = np.arange(validation_end, test_end)
        test_indices = np.arange(test_end, total_samples)
    else:
        # If not chronological, shuffle and split the indices randomly
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        test_end = int(total_samples * test_size)
        validation_end = test_end + int(total_samples * validation_size)
        
        test_indices = indices[:test_end]
        validation_indices = indices[test_end:validation_end]
        train_indices = indices[validation_end:]
        
    return train_indices, validation_indices, test_indices


def split_data_with_indices(processed_data, train_indices, validation_indices, test_indices, target_col_indices):
    """
    Split data into training, validation, and testing sets using provided indices and multiple target columns.
    """
    # Extract features from all timesteps except the last one for the training set
    X_train = processed_data[train_indices, :-1, :]
    
    # Targets are from the last timestep of each sequence for the specified target columns for the training set
    y_train = processed_data[train_indices][:, -1, :][:, target_col_indices]
    
    # Perform similar operations for validation and test sets
    X_validation = processed_data[validation_indices, :-1, :]
    y_validation = processed_data[validation_indices][:, -1, :][:, target_col_indices]
    
    X_test = processed_data[test_indices, :-1, :]
    y_test = processed_data[test_indices][:, -1, :][:, target_col_indices]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Assuming processed_data is a numpy array with the shape mentioned earlier
# Assuming you've already defined test_size and validation_size

# Create split indices based on the total number of sequences
total_sequences = processed_data.shape[0]
train_indices, validation_indices, test_indices = create_split_indices(
    total_samples=total_sequences,
    test_size=0.10,
    validation_size=0.15
)

# Assuming target_col_indices = [0, 1] because 'wert' is at index 0 and 'temp' is at index 1
target_col_indices = [0,1]

# Split the data using the indices
X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = split_data_with_indices(
    processed_data=full,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

def replace_first_two_attributes(full, synthetic):
    """
    Replaces the first two attributes of each row in 'full' with values from 'synthetic'.
    
    :param full: A 3D numpy array of shape (n_samples, n_sequence, n_attributes).
    :param synthetic: A 3D numpy array of shape (n_samples, n_sequence, 2).
    :return: A 3D numpy array with updated 'full' values.
    """
    if full.shape[0] != synthetic.shape[0] or full.shape[1] != synthetic.shape[1]:
        raise ValueError("The number of samples and sequence length must match in both arrays")

    full[:, :, :2] = synthetic
    return full

full_synthetic_data = replace_first_two_attributes(full, synthetic_data)
print(full_synthetic_data.shape)

X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = split_data_with_indices(
    processed_data= full_synthetic_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)



# COMMAND ----------

print("Shape of X_real_train:", X_real_train.shape)
print("Shape of X_real_validation:", X_real_validation.shape)
print("Shape of X_real_test:", X_real_test.shape)
print("Shape of y_real_train:", y_real_train.shape)
print("Shape of y_real_validation:", y_real_validation.shape)
print("Shape of y_real_test:", y_real_test.shape)

# COMMAND ----------

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# COMMAND ----------

# # Initialize the model
timesteps = seq_len - 1
units = 2

real_model = CNN_GRU_regression_cyclic(timesteps, 7, units)
real_model.summary()

# Train the model on real

history = real_model.fit(
    X_real_train, y_real_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# # Initialize the model
synth_model = CNN_GRU_regression_cyclic(timesteps, 7, units)

# Train the model on real
print(X_synth_train.shape, y_synth_train.shape)
print(X_real_validation.shape, y_real_validation.shape)


history = synth_model.fit(
    X_synth_train, y_synth_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# COMMAND ----------

# Split the data using the indices
full_data_seasonality = replace_first_two_attributes(full, processed_data)

X_real_train_seasonal, X_real_validation_seasonal, X_real_test_seasonal, y_real_train_seasonal, y_real_validation_seasonal, y_real_test_seasonal = split_data_with_indices(
    processed_data=full_data_seasonality,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

full_synthetic_data_seasonality = replace_first_two_attributes(full, Seasonality_synth)

X_synth_train_seasonal, X_synth_validation_seasonal, X_synth_test_seasonal, y_synth_train_seasonal, y_synth_validation_seasonal, y_synth_test__seasonal = split_data_with_indices(
    processed_data=full_synthetic_data_seasonality,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

# COMMAND ----------

print("Shape of X_real_train:", X_real_train_seasonal.shape)
print("Shape of X_real_validation:", X_real_validation_seasonal.shape)
print("Shape of X_real_test:", X_real_test_seasonal.shape)
print("Shape of y_real_train:", y_real_train_seasonal.shape)
print("Shape of y_real_validation:", y_real_validation_seasonal.shape)
print("Shape of y_real_test:", y_real_test_seasonal.shape)

# COMMAND ----------

# # Initialize the model
timesteps = seq_len - 1
units = 2

real_model_seasonal = CNN_GRU_regression_cyclic(timesteps, 7, units)

# Train the model on real

history = real_model_seasonal.fit(
    X_real_train_seasonal, y_real_train_seasonal,
    validation_data=(X_real_validation_seasonal, y_real_validation_seasonal),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# # Initialize the model
synth_model_seasonal = CNN_GRU_regression_cyclic(timesteps, 7, units)

# Train the model on real

history = synth_model_seasonal.fit(
    X_synth_train_seasonal, y_synth_train_seasonal,
    validation_data=(X_real_validation_seasonal, y_real_validation_seasonal),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# COMMAND ----------

# Generate predictions for CNN_GRU_regression
print('for real')
real_pred = real_model.predict(X_real_test)
print('for synth')
synth_pred = synth_model.predict(X_real_test)

print('Seasonal: ')
print('for real')
real_pred_seasonal = real_model_seasonal.predict(X_real_test)
print('for synth')
synthetic_pred_seasonal = synth_model_seasonal.predict(X_real_test)

# COMMAND ----------

print('compared with synthetic')
print(calculate_metrics_2(y_real_test, real_pred, synth_pred))
print('compared synthetic seasonal')
print(calculate_metrics_2(y_real_test_seasonal, real_pred_seasonal, synthetic_pred_seasonal))

# COMMAND ----------

# Call the function with your data
plot_predictions_with_real_and_synthetic(df_2023, X_real_test, y_real_test, real_pred, synth_pred, scalers, window_size=168)

# please upload to lakehouse

# COMMAND ----------

# Call the function with your data
plot_predictions_interactive(df_2023, X_real_test, y_real_test, real_pred, synth_pred, scalers, window_size=168)

# COMMAND ----------

real_model_2 = CNN_GRU_regression_cyclic_2(timesteps, 7, 1)

# Train the model on real

history = real_model_2.fit(
    X_real_train, y_real_train[:, 0],
    validation_data=(X_real_validation, y_real_validation[:, 0]),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# # Initialize the model
synth_model_2 = CNN_GRU_regression_cyclic_2(timesteps, 7, 1)

# Train the model on real
print(X_synth_train.shape, y_synth_train.shape)
print(X_real_validation.shape, y_real_validation.shape)

# COMMAND ----------

history = synth_model_2.fit(
    X_synth_train, y_synth_train[:, 0],
    validation_data=(X_real_validation, y_real_validation[:, 0]),
    epochs=100,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# COMMAND ----------

print('for real')
real_pred_2 = real_model_2.predict(X_real_test)
print(real_pred_2.shape)
print('for synth')
synth_pred_2 = synth_model_2.predict(X_real_test)
print(synth_pred_2.shape)
print('compared with synthetic')
print(calculate_metrics_2(y_real_test[:, 0].reshape(-1, 1), real_pred_2, synth_pred_2))

# COMMAND ----------

# Call the function with your data
plot_predictions_with_real_and_synthetic(df_2023, X_real_test, y_real_test, real_pred_2, synth_pred_2, scalers, window_size=168)

# COMMAND ----------

plot_predictions_interactive(df_2023, X_real_test, y_real_test, real_pred_2, synth_pred_2, scalers, window_size=168)

# COMMAND ----------

# MAGIC %md
# MAGIC # add test predicting using only wert and temp

# COMMAND ----------

# # Initialize the model
# seq_len = 24*7
# timesteps = seq_len - 1
# units = 2

# real_model = CNN_GRU_regression(timesteps, feature_n, units)

# # Train the model on real

# history = real_model.fit(
#     X_real_train, y_real_train,
#     validation_data=(X_real_validation, y_real_validation),
#     epochs=10,  # Adjust the number of epochs as needed
#     batch_size=128,  # Adjust the batch size as needed
#     callbacks=[early_stopping]
# )

# # Initialize the model
# synth_model = CNN_GRU_regression(timesteps, feature_n, units)

# # Train the model
# synth_history = synth_model.fit(
#     X_synth_train, y_synth_train,
#     validation_data=(X_real_validation, y_real_validation),
#     epochs=10,  # Adjust the number of epochs as needed
#     batch_size=128,  # Adjust the batch size as needed
#     callbacks=[early_stopping]
# )

# # Generate predictions for CNN_GRU_regression
# print('for real')
# real_pred = real_model.predict(X_real_test)
# print('for synth')
# synthetic_pred = synth_model.predict(X_real_test)

# print('compared with synthetic')
# print(calculate_metrics_2(y_real_test, real_pred, synthetic_pred))

# COMMAND ----------

# df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
# df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
## Day of the Week: There are 7 days in a week, so the transformation would be similar:

# df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
# df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
## Time of Year: Assuming you’re using day of the year for this feature, which can range from 1 to 365 or 366:

# df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear']/365.25)
# df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear']/365.25)

# COMMAND ----------

# procesded_seasonal = reseasonalize_data(processed_data, seasonal_data, seq_len, column_names)

# COMMAND ----------

# synthetic_seasonal = reseasonalize_data(synthetic_data, seasonal_data, seq_len, column_names)

# COMMAND ----------

# target = 'all_features'
# seq_len = 24

# # Split the real dataset
# X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = prepare_and_split_data_for_model(
#     procesded_seasonal, seq_len, target_type=target)

# # Split the synthetic dataset
# X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = prepare_and_split_data_for_model(
#     synthetic_seasonal, seq_len, target_type=target)

# COMMAND ----------

# # Initialize the model
# timesteps = seq_len - 1
# units = 2

# real_model = CNN_GRU_regression(timesteps, feature_n, units)

# # Train the model on real

# history = real_model.fit(
#     X_real_train, y_real_train,
#     validation_data=(X_real_validation, y_real_validation),
#     epochs=10,  # Adjust the number of epochs as needed
#     batch_size=128,  # Adjust the batch size as needed
#     callbacks=[early_stopping]
# )

# COMMAND ----------

# # Initialize the model
# synth_model = CNN_GRU_regression(timesteps, feature_n, units)

# # Train the model
# synth_history = synth_model.fit(
#     X_synth_train, y_synth_train,
#     validation_data=(X_real_validation, y_real_validation),
#     epochs=10,  # Adjust the number of epochs as needed
#     batch_size=128,  # Adjust the batch size as needed
#     callbacks=[early_stopping]
# )

# COMMAND ----------

# # Generate predictions for CNN_GRU_regression
# print('for real')
# real_pred = real_model.predict(X_real_test)
# print('for synth')
# synthetic_pred = synth_model.predict(X_real_test)

# COMMAND ----------

# print('compared with synthetic')
# print(calculate_metrics_2(y_real_test, real_pred, synthetic_pred))
