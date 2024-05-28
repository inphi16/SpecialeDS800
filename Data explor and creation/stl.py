# Databricks notebook source
import os
import sys
import io
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.seasonal import STL, DecomposeResult, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mltable import from_delta_lake


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

url = "abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/mw08003_complete_extended"

tl = from_delta_lake(url)

display(tl)

# COMMAND ----------

df = tl.to_pandas_dataframe()
df.rename(columns = {'Zeitpunkt_x': 'Zeitpunkt'}, inplace=True)
df.sort_values('Zeitpunkt', inplace= True)
df['Wert'] = df['Wert'].astype(float)

df.reset_index(drop=True, inplace=True)

# Filter the rows where the year in 'Zeitpunkt' 
df_ = df[df['Zeitpunkt'].dt.year >= 2022]

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform 'Wert' column; reshape is needed as it expects 2D array
df_['Wert_normalized'] = scaler.fit_transform(df_['Wert'].values.reshape(-1, 1))

# Your DataFrame df_ now includes a normalized 'Wert' column named 'Wert_normalized'


# COMMAND ----------

df_.head()

# COMMAND ----------

df = df_[['Zeitpunkt', 'Wert_normalized']]

# COMMAND ----------

# Function to remove seasonality and optionally plot the autocorrelation
def remove_seasonality_and_plot_acf(df, timestamp_col, value_col, period, a_lags, plot_decomposition = False, plot_acf_deseasonalized = False, plot_pacf_deseasonalized = False):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df.loc[:, timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set the timestamp column as the index
    df.set_index(timestamp_col, inplace=True)
    
    # Decompose the time series with STL
    stl = STL(df[value_col], period=period, robust=True)
    result = stl.fit()
    
    # Add the deseasonalized and seasonal components to the DataFrame
    df.loc[:, 'deseasonalized_wert'] = df[value_col] - result.seasonal
    df.loc[:, f'{period}seasonality'] = result.seasonal
    
    # Plot the decomposition if requested
    if plot_decomposition:
        result.plot()
        plt.show()
    
    # Plot the autocorrelation for deseasonalized data
    if plot_acf_deseasonalized:
        plot_acf(df["deseasonalized_wert"], lags=a_lags)  # Adjust 'lags' as needed
        plt.show()
    
    if plot_pacf_deseasonalized:
        # Plot the PACF
        plot_pacf(df["deseasonalized_wert"], lags=a_lags)  # Adjust 'lags' to determine the range to be plotted
        plt.show() 
    
    df.reset_index(inplace=True)

    return df


# COMMAND ----------

# Function to convert matplotlib plots to base64 for interactive use with plotly
plt.rcParams["figure.figsize"] = (25, 10)  # Set the default figsize

def plt_to_base64(plt_figure):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format="png")
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# Function to remove seasonality and plot interactively
def remove_seasonality_and_plot_interactive(
    df, timestamp_col, value_col, period, a_lags, plot_decomposition=False, plot_acf_deseasonalized=False, plot_pacf_deseasonalized=False
):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df.loc[:, timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set the timestamp column as the index
    df.set_index(timestamp_col, inplace=True)
    
    # Decompose the time series with STL
    from statsmodels.tsa.seasonal import STL
    stl = STL(df[value_col], period=period, robust=True)
    result = stl.fit()
    
    # Add deseasonalized and seasonal components
    df.loc[:, 'deseasonalized_wert'] = df[value_col] - result.seasonal
    df.loc[:, f'{period}seasonality'] = result.seasonal
    
    # Interactive decomposition plot
    if plot_decomposition:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        fig.add_trace(go.Scatter(x=df.index, y=df[value_col], mode="lines", name="Observed"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode="lines", name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode="lines", name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode="lines", name="Residual"), row=4, col=1)
        fig.update_layout(height=800, showlegend=True)
        fig.show()

   # Interactive ACF plot (ensure correct layout and width)
    if plot_acf_deseasonalized:
        plt.figure(figsize=(25, 10))  # Set a wider aspect ratio
        acf_plot = plot_acf(df["deseasonalized_wert"], lags=a_lags, alpha=None)
        acf_img = plt_to_base64(acf_plot.figure)
        acf_fig = go.Figure(go.Image(source=acf_img))
        acf_fig.update_layout(title="Autocorrelation", width=1200)  # Ensure correct layout and width
        acf_fig.show()  # Display the ACF plot
        plt.close(acf_plot.figure)  # Close to avoid double axes
    
    # Interactive PACF plot (ensure correct layout and width)
    if plot_pacf_deseasonalized:
        plt.figure(figsize=(25, 10))  # Set a wider aspect ratio
        pacf_plot = plot_pacf(df["deseasonalized_wert"], lags=a_lags, alpha=None)
        pacf_img = plt_to_base64(pacf_plot.figure)
        pacf_fig = go.Figure(go.Image(source=pacf_img))
        pacf_fig.update_layout(title="Partial Autocorrelation", width=1200)  # Ensure correct layout and width
        pacf_fig.show()  # Display the PACF plot
        plt.close(pacf_plot.figure)  # Close to avoid double axes
    
    
    df.reset_index(inplace=True)

    return df


# COMMAND ----------

# df2 = df[['Zeitpunkt', 'Wert']]

# COMMAND ----------

# MAGIC %md
# MAGIC 1) First to identify any seasonality we study the autocorrelation

# COMMAND ----------

# from statsmodels.graphics.tsaplots import plot_acf

# # Plot the autocorrelation
# plot_acf(df2["Wert"], lags=50)  # Adjust 'lags' to see more or less detail
# plt.show()

# COMMAND ----------

# from statsmodels.graphics.tsaplots import plot_pacf
# # Plot the PACF
# plot_pacf(df2["Wert"], lags=50)  # Adjust 'lags' to determine the range to be plotted
# plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC There appear to be peaks approximately every 12 and 24 lags. This suggests that there's a recurring pattern or seasonality with a periodicity of 24. - a peak every 12 lags could suggest a pattern that repeats every half-day.

# COMMAND ----------

# decomposed_df = remove_seasonality_and_plot_interactive(df2, "Zeitpunkt", "Wert", period=24, a_lags=504, plot_decomposition=True, plot_acf_deseasonalized=True, plot_pacf_deseasonalized = True)

# COMMAND ----------

df.head()

# COMMAND ----------

decomposed_df = remove_seasonality_and_plot_acf(df, "Zeitpunkt", "Wert_normalized", period=24, a_lags=504, plot_decomposition=True, plot_acf_deseasonalized=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Confirming Complete Seasonality Removal: By plotting the autocorrelation of the deseasonalized data, you can check whether the initial decomposition removed all significant seasonal patterns.
# MAGIC Identifying Residual Patterns: If autocorrelation still shows peaks, it indicates that some residual patterns or seasonality remain, suggesting further decomposition or analysis might be needed.

# COMMAND ----------

decomposed_df_2 = remove_seasonality_and_plot_acf(decomposed_df, "Zeitpunkt", "deseasonalized_wert", period=168, a_lags = len(df)-1, plot_decomposition=True, plot_acf_deseasonalized=True)

# Check the resulting DataFrame
decomposed_df_2.head()  # Display the first few rows of the new DataFrame

# COMMAND ----------

decomposed_df_3 = remove_seasonality_and_plot_acf(decomposed_df_2, "Zeitpunkt", "deseasonalized_wert", period=2920, a_lags = len(df)-1, plot_decomposition=True, plot_acf_deseasonalized=True)

decomposed_df_3.head()

# COMMAND ----------

spark_df = spark.createDataFrame(decomposed_df_3)

from databricks.feature_store import FeatureStoreClient
# Initialize the feature store client
fs = FeatureStoreClient()

# COMMAND ----------

# import os
# import sys
# # Define the directory
# parent_directory = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/')
# result_directory = os.path.join(parent_directory, 'results')

# # Specify the filename
# filename = 'my_data.csv'
# full_file_path = os.path.join(result_directory, filename)

# # Save the DataFrame to CSV
# pandas_df.to_csv(full_file_path, index=False)

# COMMAND ----------

# Define feature table
fs.create_table(
    name='stl_normalized', # specify which database to use with database_name.feature_table - default database name is "default"
    primary_keys=['Zeitpunkt'], 
    df=spark_df,
    description='Test - stl decomposition'
)

# Write the features to the feature store
fs.write_table(
    name='stl_normalized',
    df=spark_df,
    mode='overwrite'  # Use 'overwrite' to replace existing features or 'merge' to update
)

# COMMAND ----------

# We can then load the feature tabel with the following command
spark_df2 = fs.read_table("stl_normalized")
spark_df2

# COMMAND ----------

import os
import sys
from databricks.feature_store import FeatureStoreClient
# Initialize the feature store client
fs = FeatureStoreClient()

# Define the directory
parent_dir = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/')

# We can then load the feature tabel with the following command
# spark_df2 = fs.read_table("stl_wert_2")

# pandas_df = spark_df2.toPandas()

# Specify the filename
filename = 'stl_normalized.csv'
full_file_path = os.path.join(parent_dir, filename)

# Save the DataFrame to CSV
df_.to_csv(full_file_path, index=False)
