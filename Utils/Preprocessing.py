import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bernoulli
import random
import threadpoolctl
from statsmodels.tsa.seasonal import STL

### Classic TimeGan ### -----------
def classic_preprocess_for_timegan(df, timestamp_col, num_cols, seq_len, cat_cols=None, simulate_bernoulli=False, shuffle=False):
"""
Input:
    df: Input DataFrame containing time series data.
    timestamp_col: Name of the column containing timestamps.
    num_cols: List of names of numerical columns to be normalized.
    seq_len: Length of sequences to be created from the data.
    cat_cols: Optional list of names of categorical columns.
    simulate_bernoulli: Boll indicating whether to simulate categorical data using the Bernoulli distribution.
    shuffle: Boll indicating whether to shuffle the sequences.

It ensures the DataFrame is sorted by timestamp, normalizes the numerical columns using MinMaxScaler, and optionally simulates categorical data using the Bernoulli distribution.
Finally, it creates sequences of specified length from the data and returns them along with the scaler object and column names.
"""
    # Ensure the DataFrame is sorted by timestamp
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # Normalize the specified continuous columns
    scaler = MinMaxScaler()
    df_sorted[num_cols] = scaler.fit_transform(df_sorted[num_cols])
    
    # Initialize cat_cols if None is provided
    if cat_cols is None:
        cat_cols = []
    
    # Simulate each categorical column using Bernoulli distribution if simulate_bernoulli is True
    if simulate_bernoulli and cat_cols:
        for cat_col in cat_cols:
            p = df_sorted[cat_col].mean()  # Probability of success (1) for each column
            df_sorted[cat_col] = bernoulli.rvs(p, size=df_sorted.shape[0])

    # Concatenate column names for continuous and categorical data
    column_names_combined = num_cols + cat_cols

    # Create sequences
    sequences = []
    for i in range(len(df_sorted) - seq_len + 1):
        sequence = df_sorted[column_names_combined].iloc[i:i + seq_len].values
        sequences.append(sequence)

    # Optionally shuffle sequences
    if shuffle:
        random.shuffle(sequences)

    # Convert list of sequences to a 3D numpy array
    sequences_array = np.array(sequences)

    return sequences_array, scaler, column_names_combined


### Preprocessing remove period using STL ### ------

def preprocess_and_remove_period(df, timestamp_col, num_cols, period, seq_len=24, shuffle=False):
    # Ensure the DataFrame is sorted by timestamp
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)

    # Convert timestamp column to datetime if not already in that format
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
        df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col])

    scalers = {}
    seasonal_components = {col: [] for col in num_cols}
    sequences_array = []
    sequences_full = []

    for col in df_sorted.columns:
        if col != timestamp_col:
            scaler = MinMaxScaler()
            df_sorted[col] = scaler.fit_transform(df_sorted[[col]].values.reshape(-1, 1))
            scalers[col] = scaler

            if col in num_cols:
                col_period = period.get(col, None)
                if col_period is None:
                    raise ValueError(f"No period specified for column '{col}'")

                if not isinstance(col_period, int) or col_period < 2:
                    raise ValueError(f"Period must be a positive integer >= 2, got {col_period} for column '{col}'")

                stl = STL(df_sorted[col], period=col_period, robust=True)
                result = stl.fit()
                seasonal_component = result.seasonal
                df_sorted[col] = df_sorted[col] - seasonal_component

                # Store the seasonal components for each time slice
                for i in range(len(df_sorted) - seq_len + 1):
                    seasonal_components[col].append(seasonal_component[i:i+seq_len])

    # Create sequences
    for i in range(len(df_sorted) - seq_len + 1):
        sequence_slice = df_sorted.iloc[i:i+seq_len]
        deseasonalized_sequence = np.stack([sequence_slice[col].values for col in num_cols], axis=-1)
        sequences_array.append(deseasonalized_sequence)

        full_sequence = np.stack([sequence_slice[col].values for col in df_sorted.columns if col != timestamp_col], axis=-1)
        sequences_full.append(full_sequence)

    # Optionally shuffle sequences
    if shuffle:
        np.random.shuffle(sequences_array)
        np.random.shuffle(sequences_full)
        for col in seasonal_components:
            np.random.shuffle(seasonal_components[col])

    # Convert lists of sequences to 3D numpy arrays
    sequences_array = np.array(sequences_array)
    sequences_full = np.array(sequences_full)

    return sequences_array, sequences_full, seasonal_components, scalers

