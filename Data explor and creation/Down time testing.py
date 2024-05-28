# Databricks notebook source
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import random

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

# MAGIC %md
# MAGIC Biggest shift in temperature

# COMMAND ----------

# Convert 'Zeitpunkt' column to datetime if it's not already in datetime format
df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'])

# Filter data for the year 2024
df_2023 = df[(df['Zeitpunkt'].dt.year == 2023)].copy()  # Make a copy of the DataFrame to avoid warnings

# Calculate temperature differences between consecutive weeks
df_2023['week'] = df_2023['Zeitpunkt'].dt.isocalendar().week
df_2023['temp_diff'] = df_2023.groupby('week')['temp'].diff()

# Find the top 100 weeks with the biggest shifts in temperature
top_100_shifts = df_2023.groupby('week')['temp_diff'].sum().sort_values(ascending=False).head(100)

# Initialize a list to store results
results = []

# Iterate over top 100 weeks
for week in top_100_shifts.index:
    # Get timestamp for the start date of the week
    start_date = df_2023[df_2023['week'] == week]['Zeitpunkt'].iloc[0]
    
    # Ensure that end date is exactly 7 days (1 week) after start date
    end_date = start_date + pd.Timedelta(days=7)
    
    # Calculate temperature shift
    temp_shift = top_100_shifts.loc[week]
    
    # Append results to the list
    results.append({'Week': week, 'Start Date': start_date, 'End Date': end_date, 'Temperature Shift': temp_shift})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print the results
# print(results_df)


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
    end_date = start_date + pd.Timedelta(days=7)
    
    # Calculate temperature shift (use absolute value)
    temp_shift = top_100_shifts.loc[week]
    
    # Append results to the list
    results.append({'Week': week, 'Start Date': start_date, 'End Date': end_date, 'Temperature Shift': temp_shift})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print the results
print(results_df)


# COMMAND ----------

# Find the biggest shift
biggest_shift = results_df.nlargest(1, 'Temperature Shift')

# Find 3 random shifts
random_shifts = results_df.sample(n=3)

# Concatenate the biggest shift and 3 random shifts
selected_shifts = pd.concat([biggest_shift, random_shifts])

# Print the selected shifts
print(selected_shifts)


# COMMAND ----------

def plot_wert_for_aligned_intervals(df, selected_shifts):
    num_plots = len(selected_shifts)
    fig, axs = plt.subplots(num_plots, figsize=(10, 5 * num_plots), sharex=False)  # Each plot can have a different x-axis
    
    if num_plots == 1:
        axs = [axs]

    for i, (index, row) in enumerate(selected_shifts.iterrows()):
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        prior_week_start = start_date - pd.Timedelta(days=7)

        # Get data for current week and the week prior
        mask_current = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)

        current_week_data = df.loc[mask_current]
        prior_week_data = df.loc[mask_prior]

        # Adjust prior week data's 'Zeitpunkt' to align on the plot
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        # Plotting both datasets with aligned x-axis
        axs[i].plot(current_week_data['Zeitpunkt'], current_week_data['Wert'], label='Current Week Wert', marker='o', linestyle='-', color='green')
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='--', marker='o', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Set x-axis limits to current week range
        axs[i].set_xlim([start_date, end_date])
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()

# Call the function with your DataFrame and the selected shifts
plot_wert_for_aligned_intervals(df, selected_shifts)


# COMMAND ----------

# MAGIC %md
# MAGIC Biggest differnece in gas flow

# COMMAND ----------

df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'])

df_2023 = df[df['Zeitpunkt'].dt.year == 2023].copy()

# Calculate differences in 'Wert' between consecutive data points
df_2023 = df_2023.sort_values('Zeitpunkt')  # Ensure data is sorted by date
df_2023['wert_diff'] = df_2023['Wert'].diff().abs()  # Calculate the absolute difference

# Group by week and find the maximum 'wert_diff' to identify significant changes
df_2023['week'] = df_2023['Zeitpunkt'].dt.isocalendar().week
top_100_shifts_wert = df_2023.groupby('week')['wert_diff'].max().nlargest(100)  # Find the top 100 weeks with the largest changes

results = []

# Iterate over top 100 weeks by biggest changes in 'Wert'
for week, max_diff in top_100_shifts_wert.items():
    week_data = df_2023[df_2023['week'] == week]
    start_date = week_data['Zeitpunkt'].min()
    end_date = start_date + pd.Timedelta(days=7)

    results.append({
        'Week': week,
        'Start Date': start_date,
        'End Date': end_date,
        'Wert Shift': max_diff  
    })

# Convert results to DataFrame and sort by the magnitude of 'Wert Shift'
results_df = pd.DataFrame(results).sort_values(by='Wert Shift', ascending=False)

# Optionally, choose the biggest shift or a random sample for analysis
biggest_shift = results_df.nlargest(1, 'Wert Shift')
random_shifts = results_df.sample(n=3)

# Concatenate the biggest shift and 3 random shifts for detailed analysis
selected_shifts = pd.concat([biggest_shift, random_shifts])
print(selected_shifts)


# COMMAND ----------

def plot_wert_for_aligned_intervals(df, selected_shifts):
    num_plots = len(selected_shifts)
    fig, axs = plt.subplots(num_plots, figsize=(10, 5 * num_plots), sharex=False)  # Sharex is False to customize each x-axis
    
    if num_plots == 1:
        axs = [axs]

    for i, (index, row) in enumerate(selected_shifts.iterrows()):
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        prior_week_start = start_date - pd.Timedelta(days=7)

        # Get data for current week and the week prior
        mask_current = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)

        current_week_data = df.loc[mask_current]
        prior_week_data = df.loc[mask_prior]

        # print('n: ', len(prior_week_data))

        # Adjust prior week data's 'Zeitpunkt' to align on the plot
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        # Plotting both datasets with aligned x-axis
        axs[i].plot(current_week_data['Zeitpunkt'], current_week_data['Wert'], label='Current Week Wert', marker='o', linestyle='-', color='green')
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='--', marker='o', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Set x-axis limits to current week range
        axs[i].set_xlim([start_date, end_date])
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()


# Call the function with your DataFrame and the selected shifts
plot_wert_for_aligned_intervals(df, selected_shifts)


# COMMAND ----------

# MAGIC %md
# MAGIC Biggest average shift in gas flow

# COMMAND ----------

# Assuming 'df' is already loaded with 'Zeitpunkt' and 'Wert'
df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'])

# Sort the DataFrame by date to ensure correct difference calculations
df = df.sort_values('Zeitpunkt')

# Calculate daily differences in 'Wert'
df['wert_diff'] = df['Wert'].diff().abs()

# Extract year and week number for grouping
df['year'] = df['Zeitpunkt'].dt.year
df['week'] = df['Zeitpunkt'].dt.isocalendar().week

# Filter data for a specific year if needed, e.g., 2023
df_2023 = df[df['year'] == 2023]

# Calculate the average 'Wert' shift per week
average_weekly_shifts = df_2023.groupby('week')['wert_diff'].mean().nlargest(100)

# Store results
results = []

for week, avg_shift in average_weekly_shifts.items():
    week_data = df_2023[df_2023['week'] == week]
    start_date = week_data['Zeitpunkt'].min()
    end_date = start_date + pd.Timedelta(days=7)  # Cover the entire week

    results.append({
        'Week': week,
        'Start Date': start_date,
        'End Date': end_date,
        'Average Weekly Shift': avg_shift
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).sort_values(by='Average Weekly Shift', ascending=False)
print(results_df.head(4))  


# COMMAND ----------

def plot_average_wert_shifts(df, results_df):
    # Determine the number of plots
    num_plots = len(results_df)
    fig, axs = plt.subplots(num_plots, figsize=(10, 5 * num_plots), sharex=False)
    
    if num_plots == 1:
        axs = [axs]

    for i, row in results_df.iterrows():
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        prior_week_start = start_date - pd.Timedelta(days=7)

        # Filter data for the current week and the prior week
        mask_current = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)

        current_week_data = df.loc[mask_current]
        prior_week_data = df.loc[mask_prior]

        # Adjust prior week data's 'Zeitpunkt' to align on the plot
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        # Plotting both datasets with aligned x-axis
        axs[i].plot(current_week_data['Zeitpunkt'], current_week_data['Wert'], label='Current Week Wert', marker='o', linestyle='-', color='green')
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='--', marker='o', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Set x-axis limits to current week range
        axs[i].set_xlim([start_date, end_date])
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()

# Call the function with your DataFrame and the selected shifts
plot_average_wert_shifts(df, results_df.head(4))  # Adjust the number of rows as needed


# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction

# COMMAND ----------

from utils.Data_splitting import create_split_indices, split_data_with_indices
from utils.Prediction_models import CNN_GRU_regression
from utils.sequencer import RollingWindow
from utils.normalizer import Scaling
from utils.preprocessor import Preprocessor

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

# COMMAND ----------

scaled_df = df[['Zeitpunkt']].copy() 
 # Normalize 'Wert' column and add it to scaled_df
scaled_df[['Wert', 'temp']] = scaler.normalize(df[['Wert', 'temp']]) 


# COMMAND ----------

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

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Initialize the model
timesteps = seq_length - 1
units = len(attributes)
train_epoch = 1000

real_model = CNN_GRU_regression(timesteps, units, len(target_col_indices))

# Train the model on real

history = real_model.fit(
    X_real_train, y_real_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
    )

# COMMAND ----------

def predict_next_hour_wert(df, model, selected_shifts):
    predictions = {}

    for index, row in selected_shifts.iterrows():
        start_date = pd.to_datetime(row['Start Date'])
        prior_week_start = start_date - pd.Timedelta(days=7, hours = -1)
        prior_week_end = start_date - pd.Timedelta(hours=1)

        print(f'missing observation at {start_date}, prior week start {prior_week_start}, prior week end {prior_week_end}')

        # Filter prior week data
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
        prior_week_data = df.loc[mask_prior]

        # Check if we have enough data points
        if len(prior_week_data) != 167:  
            print(f"Not correct number of data points for prediction starting {start_date.date()}. Expected: 167, got: {len(prior_week_data)}")
            continue

        # Prepare the data for the model
        # Trimming the last point if there are 168 to match the model's expected input of 167
        input_data = prior_week_data['Wert'].to_numpy().reshape(1, 167, 1)

        # Predict using the model
        prediction = model.predict(input_data)
        predictions[start_date] = prediction.flatten()  # Store or process the prediction

    return predictions

# # Assuming you have loaded your model and data correctly
# model_predictions = predict_next_hour_wert(scaled_df, real_model, selected_shifts)

# # Print out the predictions
# for start_date, prediction in model_predictions.items():
#     print(f"Prediction for hour starting {start_date}: {prediction}")


# COMMAND ----------

# MAGIC %md
# MAGIC prediction multiple steps ahead

# COMMAND ----------

def update_sequence(sequence, new_value):
    sequence = np.roll(sequence, -1)
    sequence[0, -1] = new_value
    return sequence

# COMMAND ----------

def recursive_forecast(df, model, selected_shifts, steps=10):
    # Dictionary to store all predictions
    all_predictions = {}

    for _, row in selected_shifts.iterrows():
        start_date = pd.to_datetime(row['Start Date'])
        print('start_date:', start_date)
        
        # Store the predictions for each week
        weekly_predictions = []

        # Initial input sequence using data from the prior week
        current_input = get_prior_week_data(df, start_date)

        if current_input is None:
            print(f"Not enough data points for starting prediction at {start_date.date()}")
            all_predictions[start_date] = None
            continue
        
        for s in range(steps):
            # print('step: ', s)
            # Make prediction using the defined single-point prediction function
            next_point_prediction = model.predict(current_input)
            next_point_value = next_point_prediction[0]  # Assuming the output is an array with one element
            # Append prediction
            weekly_predictions.append(next_point_value)

            # print('current sequence', current_input)
            # print('next point', next_point_value)

            # Update the current input for the next prediction
            current_input = update_sequence(current_input, next_point_value)

        # Store weekly predictions
        all_predictions[start_date] = weekly_predictions

    return all_predictions

def get_prior_week_data(df, start_date):
    """ Retrieves and prepares the prior week's data as input for the first prediction. """
    prior_week_start = start_date - pd.Timedelta(days=7, hours = -1)
    mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
    prior_week_data = df.loc[mask_prior, 'Wert'].to_numpy()

    if len(prior_week_data) < 167:
        return None
    
    prior_week_data = prior_week_data.reshape(1, 167, 1)  # Prepare the shape as required by the model
    return prior_week_data

# # Usage
# model_predictions = recursive_forecast(scaled_df, real_model, selected_shifts)


# COMMAND ----------

# n = 10

# real_values = {}  # Dictionary to store real values corresponding to each prediction

# for start_date in model_predictions.keys():
#     # Get the real values for n hours starting from start_date
#     mask = (scaled_df['Zeitpunkt'] >= start_date) & (scaled_df['Zeitpunkt'] < start_date + pd.Timedelta(hours=n))
#     real_values[start_date] = scaled_df.loc[mask, 'Wert'].values

# # Now, you can compare the predictions with the real values
# for start_date, prediction in model_predictions.items():
#     real_value = real_values[start_date]
#     print(f"For hour starting {start_date}:")
#     print(f"  Predicted Value: {prediction}")
#     print(f"  Real Values: {real_value}")


# COMMAND ----------

# from sklearn.metrics import r2_score

# predicted_values = [0.23657331]
# real_values = [0.24260498]

# """
# Mean Absolute Error (MAE): 
# 0.00603
# Mean Squared Error (MSE): 
# 0.0000364
# Percentage Error: 
# 2.49%
# """

# COMMAND ----------

# from sklearn.metrics import r2_score, mean_squared_error

# r2_scores = []
# mse_scores = []

# for start_date, prediction in model_predictions.items():
#     real_value = real_values[start_date]
    
#     # Flatten the arrays for compatibility with sklearn functions
#     prediction_flat = [val[0] for val in prediction]
#     real_value_flat = real_value.flatten()
    
#     r2 = r2_score(real_value_flat, prediction_flat)
#     mse = mean_squared_error(real_value_flat, prediction_flat)
    
#     r2_scores.append(r2)
#     mse_scores.append(mse)

#     print(f"For hour starting {start_date}:")
#     print(f"  R2 Score: {r2}")
#     print(f"  Mean Squared Error: {mse}")

# # Compute overall scores
# overall_r2 = sum(r2_scores) / len(r2_scores)
# overall_mse = sum(mse_scores) / len(mse_scores)

# print("\nOverall Scores:")
# print(f"  Overall R2 Score: {overall_r2}")
# print(f"  Overall Mean Squared Error: {overall_mse}")


# COMMAND ----------

# MAGIC %md
# MAGIC Predict for all in the selected shifting interval

# COMMAND ----------

# def predict_for_time_range(df, model, selected_shifts):
#     predictions = {}

#     # Iterating through each row in the selected_shifts to determine the time interval
#     for _, row in selected_shifts.iterrows():
#         start_date = pd.to_datetime(row['Start Date'])
#         end_date = pd.to_datetime(row['End Date'])

#         # Iterate over each hour from start_date to end_date
#         for single_date in pd.date_range(start=start_date, end=end_date, freq='H'):
#             # Adjusting the start of the prior week data to collect exactly 167 hours of data
#             prior_week_start = single_date - pd.Timedelta(days=7, hours=-1)
            
#             # Ensure the prediction point is within the dataframe's range
#             if prior_week_start < df['Zeitpunkt'].min():
#                 continue
            
#             # Filter prior week data
#             mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < single_date)
#             prior_week_data = df.loc[mask_prior, 'Wert']
            
#             # Check if we have the correct amount of data points
#             if len(prior_week_data) != 167:
#                 print(f"Not correct number of data points for prediction at {single_date}. Expected: 167, got: {len(prior_week_data)}")
#                 continue
            
#             print(f'missing observation at {start_date}, prior week start {prior_week_start}')

#             # Prepare the data for the model
#             input_data = prior_week_data.to_numpy().reshape(1, 167, 1)

#             # Predict using the model
#             prediction = model.predict(input_data)
#             predictions[single_date] = prediction.flatten()  # Store or process the prediction

#     return predictions

# model_predictions = predict_for_time_range(df, real_model, selected_shifts)


# COMMAND ----------

def weekly_forecast(df, model, selected_shifts):
    # Dictionary to store all predictions
    all_predictions = {}

    for _, row in selected_shifts.iterrows():
        # Set start and end dates for each selected week
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])
        current_date = start_date

        weekly_predictions = []
        print('start_date:', start_date)

        while current_date < end_date:
            # Get prior 167-hour data for the current prediction point
            current_input = get_prior_week_data(df, current_date)

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

# weekly_predictions = weekly_forecast(scaled_df, real_model, selected_shifts)

# COMMAND ----------

# for key in weekly_predictions:
#     print(f"Length of values for {key}: {len(weekly_predictions[key])}")


# COMMAND ----------

def plot_forecasts_with_actuals(df, predictions_dict, selected_shifts):
    num_weeks = len(selected_shifts)
    fig, axs = plt.subplots(num_weeks, figsize=(10, 6 * num_weeks), sharex=False)

    if num_weeks == 1:
        axs = [axs]  # Ensure axs is iterable for a single plot scenario

    for i, (_, row) in enumerate(selected_shifts.iterrows()):
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])

        # Prepare actual data
        mask = (df['Zeitpunkt'] >= start_date) & (df['Zeitpunkt'] <= end_date)
        actual_data = df.loc[mask]

        # Prepare prediction data
        predictions = predictions_dict.get(start_date, [])
        prediction_dates = pd.date_range(start=start_date, periods=len(predictions), freq='H')

        # Plot actual data
        axs[i].plot(actual_data['Zeitpunkt'], actual_data['Wert'], label='Actual Wert', color='green', marker='o', linestyle='-')

        # Plot predictions
        axs[i].plot(prediction_dates, predictions, label='Predicted Wert', color='blue', marker='o', linestyle='--')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Format x-axis
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()

# plot_forecasts_with_actuals(scaled_df, weekly_predictions, selected_shifts)


# COMMAND ----------

def denormalize_predictions(predictions_dict, scaling_instance):
    denormalized_predictions = {}
    for key, values in predictions_dict.items():
        # Convert list of arrays into a single DataFrame
        df = pd.DataFrame({
            'Wert': np.concatenate(values).flatten()  # Flatten and concatenate arrays
        })
        # Use the denormalize method
        denormalized_df = scaling_instance.denormalize(df)
        # Store denormalized values
        denormalized_predictions[key] = denormalized_df['Wert'].values
    return denormalized_predictions

# # Call the function to denormalize predictions
# denormalized_pred = denormalize_predictions(weekly_predictions, scaler)


# COMMAND ----------

def plot_forecasts_with_actuals_and_prior(df, predictions_dict, selected_shifts):
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

        # Prepare prior week data
        mask_prior = (df['Zeitpunkt'] >= prior_week_start) & (df['Zeitpunkt'] < start_date)
        prior_week_data = df.loc[mask_prior]
        prior_week_data_aligned = prior_week_data.copy()
        prior_week_data_aligned['Zeitpunkt'] = prior_week_data['Zeitpunkt'] + pd.Timedelta(days=7)

        # Plot actual data
        axs[i].plot(actual_data['Zeitpunkt'], actual_data['Wert'], label='Actual Wert', color='green', marker='o', linestyle='-')

        # Plot predictions
        axs[i].plot(prediction_dates, predictions, label='Predicted Wert', color='blue', marker='o', linestyle='--')

        # Plot prior week data
        axs[i].plot(prior_week_data_aligned['Zeitpunkt'], prior_week_data['Wert'], label='Prior Week Wert', linestyle='--', marker='o', color='red')

        axs[i].set_title(f"Week {row['Week']} from {start_date.date()} to {end_date.date()}")
        axs[i].set_ylabel('Wert (Gas Consumption)')
        axs[i].legend()

        # Format x-axis
        axs[i].xaxis.set_major_locator(mdates.DayLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].figure.autofmt_xdate()

    plt.tight_layout()
    plt.show()


# plot_forecasts_with_actuals_and_prior(df, denormalized_pred, selected_shifts)


# COMMAND ----------

# MAGIC %md
# MAGIC Synthetic model

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

X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = split_data_with_indices(
    processed_data=processed_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

# COMMAND ----------

X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = split_data_with_indices(
    processed_data=synth_data,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    target_col_indices=target_col_indices
)

# COMMAND ----------

print(X_synth_train.shape)
print(X_synth_validation.shape)
print(y_synth_train.shape)
print(y_synth_validation.shape)
print(X_real_validation.shape)

# COMMAND ----------

# Initialize the model
attributes = ['Wert', 'temp']
units = len(attributes)

synth_model = CNN_GRU_regression(timesteps, units, len(target_col_indices))

# Train the model
synth_history = synth_model.fit(
    X_synth_train, y_synth_train,
    validation_data=(X_real_validation, y_real_validation),
    epochs=train_epoch,  # Adjust the number of epochs as needed
    batch_size=128,  # Adjust the batch size as needed
    callbacks=[early_stopping]
)

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

        while current_date < end_date:
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

scaled_df

# COMMAND ----------

weekly_synth_predictions = weekly_forecast(scaled_df, synth_model, selected_shifts, attributes)

# COMMAND ----------

def denormalize_predictions_2(predictions_dict, scaling_instance):
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

# Example usage
denormalized_synth_pred = denormalize_predictions_2(weekly_synth_predictions, scaler)
