# Databricks notebook source
import pyspark as ps
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC AND [InfoPunkt] = 'MW08382';

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Examination 

# COMMAND ----------

df = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/neptun_hlnt_archive_1hour")
df

# COMMAND ----------

relevant_stations = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/Stations")
relevant_stations

# COMMAND ----------

# As we are only interested in M/R stations
# Filter the DataFrame where 'b2T' is 'M/R'
filtered_df = relevant_stations[relevant_stations['b2T'] == 'M/R']

# Find unique values in 'InfoPunkt' in the filtered DataFrame
unique_infopunkt_mr = filtered_df['InfoPunkt'].unique()

print("Unique values in 'InfoPunkt' where 'b2T' is 'M/R':", unique_infopunkt_mr)
print(f"There are {len(unique_infopunkt_mr)} unique stations where 'b2T' is 'M/R'")


# COMMAND ----------

result_df = df[df['Infopunkt'].isin(filtered_df['InfoPunkt'])]
result_df

# COMMAND ----------

# MAGIC %md
# MAGIC undersøg 
# MAGIC MW08521    5855 (stenlille)
# MAGIC MW08261    6126 (lilballe)

# COMMAND ----------

# Counting observations for each 'InfoPunkt'
observation_count = result_df['Infopunkt'].value_counts()

# Sorting the observation count in ascending order
sorted_observations = observation_count.sort_values()

print(sorted_observations)

# COMMAND ----------

missing_data_info = {}

for station in sorted_observations.index:
    # Filter DataFrame for the current station and the columns 'InfoPunkt' and 'Zeitpunkt'
    station_df = result_df[(result_df['Infopunkt'] == station)][['Infopunkt', 'Zeitpunkt']]

    # Determine the date range for the current station
    min_date = station_df['Zeitpunkt'].min()
    max_date = station_df['Zeitpunkt'].max()

    if station == 'MW09722':
        print(min_date)
        print(max_date)

    # Set 'Zeitpunkt' as the index and resample to hourly frequency
    station_df.set_index('Zeitpunkt', inplace=True)
    hourly_data = station_df.resample('H').asfreq()

    # The resampled DataFrame will have NaNs where data is missing; we count these NaNs
    missing_data_info[station] = hourly_data['Infopunkt'].isna().sum()

missing_data_info


# COMMAND ----------

# MAGIC %md
# MAGIC # MW08003 Preprossing

# COMMAND ----------

MW08003 = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/MW08003_timeseries")
MW08003


# COMMAND ----------

min_date = MW08003['Zeitpunkt'].min()
max_date = MW08003['Zeitpunkt'].max()
print("Minimum Date:", min_date)
print("Maximum Date:", max_date)


# COMMAND ----------

### check data for completness ###
# Set 'Zeitpunkt' as the index
MW08003.set_index('Zeitpunkt', inplace=True)

# Resample to hourly frequency and check for missing data
hourly_data = MW08003.resample('H').asfreq()

# Find where the measurements are NaN
missing_data = hourly_data[hourly_data.isna().any(axis=1)]

# Report missing timestamps
print("Missing measurements at the following timestamps:")
print(missing_data.index)
print(f"There is {len(missing_data.index)} missing data points")

# COMMAND ----------

### imputation using K-Nearest Neighbors to fill in the missing values. ###
k = 5

# Create complete dataset from min to max date
complete_index = pd.date_range(start=MW08003.index.min(), end=MW08003.index.max(), freq='H', tz='UTC')

# Reindex the DataFrame using the complete index, introducing NaNs for missing timestamps
MW08003_complete = MW08003.reindex(complete_index)

# Apply forward rolling window
forward_filled = MW08003_complete['Wert'].rolling(window=k, min_periods=1).apply(lambda x: x[1:].mean(), raw=False)

# Apply backward rolling window
backward_filled = MW08003_complete['Wert'][::-1].rolling(window=k, min_periods=1).apply(lambda x: x[1:].mean(), raw=False)[::-1]

# Combine forward and backward filled values by taking their average
combined_fill = (forward_filled + backward_filled) / 2

# Replace NaNs in the original data with the combined values
MW08003_complete['Wert'] = MW08003_complete['Wert'].where(MW08003_complete['Wert'].notna(), combined_fill)

# Check if there are any NaN values left
remaining_nans = MW08003_complete['Wert'].isnull().sum()
print(f"Remaining NaN values after imputation: {remaining_nans}")

# COMMAND ----------

# MAGIC %md
# MAGIC # DMI preprocessing

# COMMAND ----------

dmi = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/dmi")
dmi

# COMMAND ----------

# Assuming dmi['from'] is a column with timestamps
dmi['from'] = pd.to_datetime(dmi['from'])
# Assuming dmi['from'] is a column with timestamps
dmi['to'] = pd.to_datetime(dmi['to'])
# rename to temp
dmi = dmi.rename(columns={"value": "temp"})
# check for missing values
missing_values = dmi.isnull().sum()
print(missing_values)

# COMMAND ----------

# MAGIC %md
# MAGIC # Final dataset

# COMMAND ----------

# Reset the index without dropping it, and rename the new column to 'Zeitpunkt'
MW08003_complete.reset_index(inplace=True)
MW08003_complete.rename(columns={'index': 'Zeitpunkt'}, inplace=True)

merged_df = pd.merge(MW08003_complete, dmi, left_on='Zeitpunkt', right_on='from', how='inner')
merged_df.drop(['id', 'to', 'from', 'parameterId', 'municipalityName'], axis=1, inplace=True)
# Convert 'Zeitpunkt' to datetime if not already done
merged_df['Zeitpunkt'] = pd.to_datetime(merged_df['Zeitpunkt'])
# Set 'Zeitpunkt' as the index
merged_df

# COMMAND ----------

merged_df.dtypes

# COMMAND ----------

# Convert 'Wert' to float64
merged_df['Wert'] = merged_df['Wert'].astype('float64')

# Now try to create the Spark DataFrame again
spark_df = spark.createDataFrame(merged_df)

# COMMAND ----------

# spark_df.write.format("delta").saveAsTable('MW08003_complete')

# COMMAND ----------

# MAGIC %md
# MAGIC # Day type prep

# COMMAND ----------

import pandas as pd
import holidays
from datetime import datetime, timedelta


# COMMAND ----------

df2 = pd.read_parquet("abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/mw08003_complete")
df2.sort_values('Zeitpunkt', inplace= True)
df2['Wert'] = df2['Wert'].astype(float)

# COMMAND ----------

# 2020-01-01 00:00:00+00:00
# 2024-02-22 11:00:00+00:00
# Function to calculate Easter Sunday based on the year
def easter_sunday(year):
    # Algorithm to calculate Easter Sunday
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)

# Function to calculate Fastelavn, which is 7 weeks before Easter
def fastelavn(year):
    return easter_sunday(year) - timedelta(weeks=7)

# Function to calculate Mother's Day, which is the second Sunday of May
def mothers_day(year):
    may_first = datetime(year, 5, 1)
    weekday = may_first.weekday()
    # If May 1st is Sunday, Mother's Day is on the 8th
    if weekday == 6:
        return may_first + timedelta(days=7)
    # Otherwise, it's the following Sunday
    else:
        return may_first + timedelta(days=(6 - weekday + 7))

# Create a range of dates for the years 2020 to 2024
start_date = "2020-01-01"
end_date = "2024-02-22"
dates = pd.date_range(start=start_date, end=end_date)

# Create a DataFrame
df = pd.DataFrame({'date': dates})
df['day'] = df['date'].dt.strftime('%a')  # Short day names

# Get Danish holidays using the holidays package for each year
dk_holidays = {}
for year in range(2020, 2025):
    for date, name in sorted(holidays.Denmark(years=year).items()):
        dk_holidays[date] = name

# Add a column for Danish holidays, mark 'holiday' if the date is in the dk_holidays dictionary
holiday_dates = [pd.Timestamp(h).floor('D') for h in dk_holidays.keys()]
df['day_type'] = df['date'].apply(lambda x: 'holiday' if x in holiday_dates else '')


# Calculate Fastelavn and Mother's Day dates
fastelavn_dates = {year: fastelavn(year) for year in range(2020, 2025)}
mothers_day_dates = {year: mothers_day(year) for year in range(2020, 2025)}

# Define additional holidays with variable dates
additional_holidays = {
    'Valentinsdag': {year: f"{year}-02-14" for year in range(2020, 2025)},
    'Fastelavn': fastelavn_dates,
    'Arbejdernes kampdag': {year: f"{year}-05-01" for year in range(2020, 2025)},
    'Mors Dag': mothers_day_dates,
    # [Continue with other holidays]
}

# Add additional holidays to the DataFrame
for holiday_name, holiday_dates in additional_holidays.items():
    for year, date in holiday_dates.items():
        df.loc[df['date'] == date, 'day_type'] = 'holiday'

# Displaying the first few rows of the DataFrame
df.head()

# COMMAND ----------

# # Fill in 'Friday' for Fridays if 'day_type' column is empty
# fri_days = ['Fri']
# df.loc[df['day'].isin(fri_days) & df['day_type'].eq(''), 'day_type'] = 'Friday'

# Fill in 'Weekend' for weekends if 'day_type' column is empty
weekend_days = ['Fri', 'Sat', 'Sun']
df.loc[df['day'].isin(weekend_days) & df['day_type'].eq(''), 'day_type'] = 'Weekend'

# Fill in 'Weekday' for weekdays if 'day_type' column is empty
week_days = ['Mon', 'Tue', 'Wed', 'Thu']
df.loc[df['day'].isin(week_days) & df['day_type'].eq(''), 'day_type'] = 'Weekday'

df['day_type_indicator'] = df['day_type'].apply(
    lambda x: 0 if x == 'Weekday' else 1 # (1 if x == 'Weekend' else 2)
    )

# # Convert day_type to a binary indicator: 0 for Weekday, 1 for Weekend, 2 for all other types
# df['day_type_indicator'] = df['day_type'].apply(
#     lambda x: 0 if x == 'Weekday' else (1 if x == 'Friday' else (2 if x == 'Weekend' else 3))
# )


# COMMAND ----------

# Convert 'Zeitpunkt' to datetime and extract the date to a new column
df2['date'] = pd.to_datetime(df2['Zeitpunkt']).dt.date

# Convert 'date' in df to datetime
df['date'] = pd.to_datetime(df['date']).dt.date

# Merge the dataframes using a left join on the date column
final_df = pd.merge(df2, df, on='date', how='left')

# COMMAND ----------

def get_season(timestamp):
    month = timestamp.month
    if 3 <= month <= 5:
        return 1  # Spring
    elif 6 <= month <= 8:
        return 2  # Summer
    elif 9 <= month <= 11:
        return 3  # Autumn
    else:
        return 4  # Winter

# Assuming df is your DataFrame and 'Zeitpunkt' is the timestamp column
final_df['season'] = final_df['Zeitpunkt'].apply(get_season)

# COMMAND ----------

# Convert 'Zeitpunkt' to datetime and set as index
final_df['Zeitpunkt'] = pd.to_datetime(final_df['Zeitpunkt'])
final_df.set_index('Zeitpunkt', inplace=True)

# Calculate daily median and standard deviation of temperature
daily_median = final_df['temp'].resample('D').median()
daily_std = final_df['temp'].resample('D').std()

# Create a DataFrame from the aggregates for easier merging
daily_stats = pd.DataFrame({'daily_median_temp': daily_median, 'daily_std_temp': daily_std})

# Merge the daily statistics into the original DataFrame by resetting the index and merging on the date
final_df.reset_index(inplace=True)
final_df['date'] = final_df['Zeitpunkt'].dt.date
daily_stats.reset_index(inplace=True)
daily_stats['date'] = daily_stats['Zeitpunkt'].dt.date

# Now merge using the 'date' column as the key
final_df = pd.merge(final_df, daily_stats, on='date', how='left')

# Drop the extra 'date' column if no longer needed
final_df.drop('date', axis=1, inplace=True)

# COMMAND ----------

final_df = final_df.drop(columns = ['day', 'Zeitpunkt_y'])
final_df['daily_std_temp'].fillna(0, inplace=True)
final_df

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

def plot_seasonal_hourly_averages(df, timestamp_col, value_col, season_col):
    """
    Plots average values by hour for each season.

    Parameters:
    - df: pandas DataFrame containing the data
    - timestamp_col: the name of the column containing the timestamp data
    - value_col: the name of the column containing the values to average
    - season_col: the name of the column containing the seasonal information
    """
    # Convert timestamp column to datetime and extract hour
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df[timestamp_col].dt.hour

    # Map the season column to actual season names
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    df['season_name'] = df[season_col].map(season_map)

    # Group by season and hour and calculate the mean
    seasonal_hourly_mean = df.groupby(['season_name', 'hour'])[value_col].mean().unstack(level=0)

    # Time of day for x-axis
    time_day = np.arange(24)

    # Plotting
    plt.figure(figsize=(15, 7))
    for season in season_map.values():
        plt.plot(time_day, seasonal_hourly_mean[season], 'o-', label=f'{season} average')
    plt.plot(time_day, seasonal_hourly_mean.mean(axis=1), 'o-', label='Total average', linewidth=2)

    plt.xlabel('Hour')
    plt.ylabel('Imported wert (nm3)')
    plt.title('Average Wert by Season and Hour')
    plt.xticks(time_day)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_seasonal_hourly_averages(final_df, 'Zeitpunkt_x', 'Wert', 'season')


# COMMAND ----------

def plot_weekly_averages(df, timestamp_col, value_col, season_col):
    """
    Plots average values by day of the week for each season.

    Parameters:
    - df: pandas DataFrame containing the data
    - timestamp_col: the name of the column containing the timestamp data
    - value_col: the name of the column containing the values to average
    - season_col: the name of the column containing the seasonal information
    """
    # Convert timestamp column to datetime and extract day of the week
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['weekday'] = df[timestamp_col].dt.day_name()

    # Map the season column to actual season names
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    df['season_name'] = df[season_col].map(season_map)

    # Group by season and day of the week and calculate the mean
    seasonal_weekly_mean = df.groupby(['season_name', 'weekday'])[value_col].mean().unstack(level=0)

    # Order the days of the week starting from Monday
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    seasonal_weekly_mean = seasonal_weekly_mean.reindex(ordered_days)

    # Plotting
    plt.figure(figsize=(15, 7))
    for season in season_map.values():
        plt.plot(ordered_days, seasonal_weekly_mean[season], 'o-', label=f'{season} average')
    plt.plot(ordered_days, seasonal_weekly_mean.mean(axis=1), 'o-', label='Total average', linewidth=2)

    plt.xlabel('Day of the Week')
    plt.ylabel('Average Wert (nm3)')
    plt.title('Average Wert by Season and Day of the Week')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust the padding to ensure everything fits without overlap
    plt.show()

plot_weekly_averages(final_df, 'Zeitpunkt_x', 'Wert', 'season')


# COMMAND ----------

# Create columns for hour and weekday
final_df['hour'] = final_df['Zeitpunkt_x'].dt.hour
final_df['weekday'] = final_df['Zeitpunkt_x'].dt.day_name()

# Map the 'season' column to season names
season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
final_df['season_name'] = final_df['season'].map(season_map)

# Calculate the hourly and weekly averages by season as DataFrames
hourly_averages_df = final_df.groupby(['season_name', 'hour'])['Wert'].mean().reset_index().rename(columns={'Wert': 'hourly_avg_wert'})
weekly_averages_df = final_df.groupby(['season_name', 'weekday'])['Wert'].mean().reset_index().rename(columns={'Wert': 'weekly_avg_wert'})

# Merge the averages back into the original DataFrame on 'season_name' and 'hour'/'weekday'
final_df = final_df.merge(hourly_averages_df, on=['season_name', 'hour'], how='left')
final_df = final_df.merge(weekly_averages_df, on=['season_name', 'weekday'], how='left')

# COMMAND ----------

final_df

# COMMAND ----------

# Now try to create the Spark DataFrame again
spark_df = spark.createDataFrame(final_df)
# spark_df.write.format("delta").saveAsTable('MW08003_complete_extended')

# COMMAND ----------

def plot_nested_season_daytype_pie(df, season_col, day_type_indicator_col):
    """
    Plots a nested pie chart with the outer ring representing seasons and 
    the inner ring representing the distribution of workdays and holidays.

    Parameters:
    - df: pandas DataFrame containing the data
    - season_col: the name of the column containing the seasonal information
    - day_type_indicator_col: the name of the column indicating workday (0) or holiday (1)
    """
    # Map season indices to names
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    df['season_name'] = df[season_col].map(season_map)
    
    # Count the number of workdays and holidays for each season
    season_counts = df.groupby('season_name')[day_type_indicator_col].value_counts().unstack().fillna(0)
    season_totals = season_counts.sum(axis=1)
    
    # Outer ring - Seasons
    colors_outer = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    fig, ax = plt.subplots()
    ax.pie(season_totals, labels=season_totals.index, radius=1, colors=colors_outer,
           wedgeprops=dict(width=0.3, edgecolor='w'), startangle=140, autopct='%1.1f%%')

    # Inner ring - Workday/Holiday
    workday_holiday_colors = ['#c2c2f0','#ffb3e6']
    workday_holiday_labels = ['Workday', 'Holiday']
    pie_inner = []
    for season in season_map.values():
        pie_inner.extend(season_counts.loc[season])
    
    ax.pie(pie_inner, labels=None, radius=0.7, colors=workday_holiday_colors*len(season_map),
           wedgeprops=dict(width=0.3, edgecolor='w'), startangle=140, autopct='%1.1f%%')

    # Draw a circle at the center to make it a donut chart
    centre_circle = plt.Circle((0,0),0.4,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.show()
    
plot_nested_season_daytype_pie(final_df, 'season', 'day_type_indicator')

# COMMAND ----------

import plotly.express as px

def plot_season_day_type_distribution(df, season_col='season', day_type_col='day_type_indicator'):
    """
    Plots a sunburst chart showing the distribution of workdays and holidays within each season.

    Parameters:
    - df: pandas DataFrame containing the data
    - season_col: the name of the column containing the seasonal information
    - day_type_col: the name of the column indicating workday (0) or holiday (1)
    """
    # Map season indices to names
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    df['season_name'] = df[season_col].map(season_map)
    
    # Map day type indices to names
    day_type_map = {0: 'Workday', 1: 'Holiday'}
    df['day_type_name'] = df[day_type_col].map(day_type_map)
    
    # Create a sunburst chart
    fig = px.sunburst(
        df, 
        path=['season_name', 'day_type_name'],  # Define hierarchy
        # values='value',  # If you have a column that aggregates, you can specify it here
        title='Proportion of Workdays and Holidays by Season'
    )
    
    # Set the visual properties
    fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
    fig.layout.font.family = "Times New Roman"
    fig.layout.width = 1000
    fig.layout.height = 1000
    
    # Show the plot
    fig.show()
    
plot_season_day_type_distribution(final_df)

