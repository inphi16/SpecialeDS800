# Databricks notebook source
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import random
from sklearn.metrics import r2_score, mean_absolute_error

# Define the path one level up
parent_directory = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/')
# Add this path to the sys.path list
sys.path.append(f"{parent_directory}")

# COMMAND ----------

# Define path to the data
file_path = os.path.join(parent_directory, 'Data')
df = pd.read_csv(f'{file_path}/mw08003.csv')
df = df[['Zeitpunkt', 'Wert', 'temp']]

# COMMAND ----------

df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'])

# Filter data for the year 2023
df_2023 = df[(df['Zeitpunkt'].dt.year == 2023)].copy()  # Make a copy of the DataFrame to avoid warnings

# Calculate temperature differences between consecutive weeks
df_2023['week'] = df_2023['Zeitpunkt'].dt.isocalendar().week
df_2023['temp_diff'] = df_2023.groupby('week')['temp'].diff()

# Find the top 100 weeks with the biggest shifts in temperature
top_100_shifts = df_2023.groupby('week')['temp_diff'].sum().abs().sort_values(ascending=False).head(100)

# Initialize a list to store results
results = []

# Iterate over top 100 weeks
for week in top_100_shifts.index:
    # Get timestamp for the start date of the week
    start_date = df_2023[df_2023['week'] == week]['Zeitpunkt'].iloc[0]
    
    # Ensure that end date is exactly 7 days (1 week) after start date
    end_date = start_date + pd.Timedelta(days=7, hours = -1)
    
    # Calculate temperature shift (use absolute value)
    temp_shift = top_100_shifts.loc[week]
    
    # Append results to the list
    results.append({'Week': week, 'Start Date': start_date, 'End Date': end_date, 'Temperature Shift': temp_shift})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print the results
print(results_df)


# COMMAND ----------

# # Find the biggest shift
# biggest_shift = results_df.nlargest(1, 'Temperature Shift')

# # Find 3 random shifts
# random_shifts = results_df.sample(n=3)

# # Concatenate the biggest shift and 3 random shifts
# selected_shifts = pd.concat([biggest_shift, random_shifts])

# # Print the selected shifts
# print(selected_shifts)

# Option 3: Select multiple specific weeks by list of Week Numbers
week_numbers = [3, 19, 4, 13]  # List of week numbers you are interested in
selected_shifts = results_df[results_df['Week'].isin(week_numbers)]

# Print the selected week(s)
print(selected_shifts)


# COMMAND ----------

# MAGIC %md # Model training
# MAGIC

# COMMAND ----------

from utils.Data_splitting import create_split_indices, split_data_with_indices
from utils.sequencer import RollingWindow
from utils.normalizer import Scaling
from utils.preprocessor import Preprocessor
from performance_metrics.Prediction_models import CNN_GRU_regression_2

# COMMAND ----------

attributes = ['Wert', 'temp'] 
feature_n = len(attributes)
seq_length = 24*7
time_col = 'Zeitpunkt'

### Initialize ###
# Initialize the Scaling class
scaler = Scaling(value_cols=attributes)

# Initialize sequencer
sequencer = RollingWindow(seq_number=seq_length, time_col=time_col, value_cols=attributes)

# Initialize preprossing
preprocessor = Preprocessor(data=df, normalizer=scaler, sequencer=sequencer)

processed_data = preprocessor.preprocess()

scaled_df = df[['Zeitpunkt']].copy() 
 # Normalize 'Wert' column and add it to scaled_df
scaled_df[['Wert', 'temp']] = scaler.normalize(df[['Wert', 'temp']]) 


# COMMAND ----------

# Initialize the model
units = len(attributes)
target_col_indices = [0]

# Create split indices based on the total number of sequences
total_sequences = processed_data.shape[0]
train_indices, validation_indices, test_indices = create_split_indices(
    total_samples=total_sequences,
    test_size=0.10,
    validation_size=0.15
)

# Split the data using the indices
X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = split_data_with_indices(
    processed_data=processed_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

train_epoch = 1000

real_model = CNN_GRU_regression_2(timesteps=167, features_per_timestep=2, units=32)

# Train the model on real

history = real_model.fit(
    X_real_train, y_real_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Synthetic

# COMMAND ----------

# Define path to synthetic data
synthetic_file_path = os.path.join(parent_directory, 'results/results_classic_168_with_temp')
synth_flat = pd.read_csv(f'{synthetic_file_path}/synthetic_data_flattened_2_2.csv', header=None)
# Reshape the flattened array back to its original shape
original_shape = (18602, 168, 2)
synth_data = synth_flat.values.reshape(original_shape)

# Check the shape of the original array
print(synth_data.shape)  

# COMMAND ----------

X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = split_data_with_indices(
    processed_data=synth_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

synth_model = CNN_GRU_regression_2(timesteps=167, features_per_timestep=2, units=32)

# Train the model
synth_history = synth_model.fit(
    X_synth_train, y_synth_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Prediction

# COMMAND ----------

def get_prior_week_data(df, start_date, attributes):
    """ Retrieves and prepares the prior week's data as input for the first prediction. """
    prior_week_start = start_date - pd.Timedelta(days=7, hours=-1)
    mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
    
    prior_week_data = df.loc[mask_prior, attributes].to_numpy()

    if len(prior_week_data) < 167:
        return None
    
    prior_week_data = prior_week_data.reshape(1, 167, len(attributes))  # Prepare the shape as required by the model
    return prior_week_data


# COMMAND ----------

def weekly_forecast(df, model, selected_shifts, attributes):
    # Dictionary to store all predictions
    all_predictions = {}

    for _, row in selected_shifts.iterrows():
        # Set start and end dates for each selected week
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        current_date = start_date

        weekly_predictions = []
        print('start_date:', start_date)

        while current_date <= end_date:
            # Get prior 167-hour data for the current prediction point
            current_input = get_prior_week_data(df, current_date, attributes)

            if current_input is None:
                print(f"Not enough data points for starting prediction at {current_date}")
                weekly_predictions.append(None)
            else:
                # Predict the next point
                next_point_prediction = model.predict(current_input)
                next_point_value = next_point_prediction[0]  # Assuming output is an array with one element

                # Append the predicted value to the weekly predictions
                weekly_predictions.append(next_point_value)

            # Move to the next hour
            current_date += pd.Timedelta(hours=1)

        # Store weekly predictions
        all_predictions[start_date] = weekly_predictions

    return all_predictions


# COMMAND ----------

weekly_predictions = weekly_forecast(scaled_df, real_model, selected_shifts, attributes)

weekly_synth_predictions = weekly_forecast(scaled_df, synth_model, selected_shifts, attributes)

# COMMAND ----------

def denormalize_predictions(predictions_dict, scaling_instance):
    denormalized_predictions = {}
    for key, values in predictions_dict.items():
        # Convert list of arrays into a single DataFrame
        df = pd.DataFrame({
            'Wert': np.concatenate(values).flatten()  # Flatten and concatenate arrays
        })
        # Create a dummy 'temp' column with zero or another appropriate default value
        df['temp'] = 0
        
        # Use the denormalize method
        denormalized_df = scaling_instance.denormalize(df[['Wert', 'temp']])
        
        # Store only denormalized 'Wert' values
        denormalized_predictions[key] = denormalized_df['Wert'].values
    return denormalized_predictions


denormalized_pred = denormalize_predictions(weekly_predictions, scaler)

denormalized_synth_pred = denormalize_predictions(weekly_synth_predictions, scaler)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization of predictions

# COMMAND ----------

def plot_forecasts_with_actuals_and_prior(df, predictions_dict, selected_shifts, denormalized_predictions_dict):
    num_weeks = len(selected_shifts)
    fig, axs = plt.subplots(num_weeks, figsize=(10, 6 * num_weeks), sharex=False)

    if num_weeks == 1:
        axs = [axs]  # Ensure axs is iterable for a single plot scenario

    for i, (_, row) in enumerate(selected_shifts.iterrows()):
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        prior_week_start = start_date - pd.Timedelta(days=7)

        # Prepare actual data
        mask_actual = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        actual_data = df.loc[mask_actual]

        # Prepare prediction data
        predictions = predictions_dict.get(start_date, [])
        prediction_dates = pd.date_range(start=start_date, periods=len(predictions), freq='H')

        # Prepare denormalized synthetic predictions data
        denormalized_predictions = denormalized_predictions_dict.get(start_date, [])
        denormalized_prediction_dates = pd.date_range(start=start_date, periods=len(denormalized_predictions), freq='H')

        # Prepare prior week data
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
        prior_week_data = df.loc[mask_prior]
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        # Plot actual data
        axs[i].plot(actual_data['Zeitpunkt'], actual_data['Wert'], label='Actual Wert', color='green', linestyle='-')

        # Plot predictions
        axs[i].plot(prediction_dates, predictions, label='Predicted Wert', color='blue', linestyle='-')

        # Plot denormalized synthetic predictions
        axs[i].plot(denormalized_prediction_dates, denormalized_predictions, label='Synthetic Predictions', color='purple', linestyle='-')

        # Plot prior week data
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='-', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Format x-axis
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()

# plot_forecasts_with_actuals_and_prior(df, denormalized_pred, selected_shifts, denormalized_synth_pred)


# COMMAND ----------

def plot_forecasts_with_actuals_and_prior(df, predictions_dict, selected_shifts, denormalized_predictions_dict):
    num_weeks = len(selected_shifts)
    fig, axs = plt.subplots(num_weeks, figsize=(14, 6 * num_weeks), sharex=False)
    score_data = []  # List to store scores for each plot

    if num_weeks == 1:
        axs = [axs]

    for i, (_, row) in enumerate(selected_shifts.iterrows()):
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        prior_week_start = start_date - pd.Timedelta(days=7)

        mask_actual = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        actual_data = df.loc[mask_actual]

        predictions = predictions_dict.get(start_date, [])
        prediction_dates = pd.date_range(start=start_date, periods=len(predictions), freq='H')

        denormalized_predictions = denormalized_predictions_dict.get(start_date, [])
        denormalized_prediction_dates = pd.date_range(start=start_date, periods=len(denormalized_predictions), freq='H')

        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
        prior_week_data = df.loc[mask_prior]
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        axs[i].plot(actual_data['Zeitpunkt'], actual_data['Wert'], label='Actual Wert', color='green', linestyle='-')
        axs[i].plot(prediction_dates, predictions, label='Predicted Wert', color='blue', linestyle='-')
        axs[i].plot(denormalized_prediction_dates, denormalized_predictions, label='Synthetic Predictions', color='purple', linestyle='-')
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='-', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

    plt.tight_layout()

    plt.savefig(f'{parent_directory}/predictions.png')

    plt.show()

# Ensure to define df, predictions_dict, selected_shifts,
plot_forecasts_with_actuals_and_prior(df, denormalized_pred, selected_shifts, denormalized_synth_pred)

# COMMAND ----------

def compute_scores(df, predictions_dict, selected_shifts, synthetic_dict):
    scores = []
    for _, row in selected_shifts.iterrows():
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        
        # Actual values
        actual_values = df.loc[(df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date), 'Wert']
        actual_index = pd.date_range(start=start_date, periods=len(actual_values), freq='H')

        # Prior week values
        prior_values = df.loc[(df['Zeitpunkt'] >= start_date - pd.Timedelta(days=7)) & (df['Zeitpunkt'] < start_date), 'Wert'].values
        if len(prior_values) == len(actual_values):
            prior_series = pd.Series(prior_values, index=actual_index)
        else:
            prior_series = pd.Series(np.full(len(actual_values), np.nan), index=actual_index)

        # Predicted values
        predicted_values = predictions_dict.get(start_date, np.full(len(actual_values), np.nan))
        predicted_series = pd.Series(predicted_values, index=actual_index)

        # Synthetic values
        synthetic_values = synthetic_dict.get(start_date, np.full(len(actual_values), np.nan))
        synthetic_series = pd.Series(synthetic_values, index=actual_index)

        # Compute scores
        r2_pred = r2_score(actual_values, predicted_series, multioutput='variance_weighted')
        mae_pred = mean_absolute_error(actual_values, predicted_series)

        r2_synth = r2_score(actual_values, synthetic_series, multioutput='variance_weighted')
        mae_synth = mean_absolute_error(actual_values, synthetic_series)

        r2_prior = r2_score(actual_values, prior_series, multioutput='variance_weighted')
        mae_prior = mean_absolute_error(actual_values, prior_series)

        # Collect scores
        scores.append({
            'Week Starting': start_date.date(),
            'R2 Score Predicted': r2_pred,
            'MAE Score Predicted': mae_pred,
            'R2 Score Synthetic': r2_synth,
            'MAE Score Synthetic': mae_synth,
            'R2 Score Prior': r2_prior,
            'MAE Score Prior': mae_prior
        })

    return pd.DataFrame(scores)

# Assuming df is your DataFrame and other variables are defined as per your context.
scores_df = compute_scores(df, denormalized_pred, selected_shifts, denormalized_synth_pred)
scores_df.display()

