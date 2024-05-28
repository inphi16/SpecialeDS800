import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def Compose(data, scaler, scaler_new, num_samples, num_time_steps, num_features, column_names):
    """
    Compose the dataset by summing up specified attributes to create a 'Wert' column,
    normalizing, and reshaping it.

    Parameters:
        data (np.array): Input data array to process.
        scaler (MinMaxScaler): Scaler to normalize data.
        num_samples (int): Number of samples in the dataset.
        num_time_steps (int): Number of time steps in the dataset.
        num_features (int): Number of features per time step.
        column_names (list): List of column names for the dataset.

    Returns:
        np.array: The composed, normalized, and reshaped dataset.
    """
    # Validate the number of features
    if data.shape[2] != len(column_names):
        raise ValueError("The number of features in the array does not match the expected number of seasonality columns.")

    # Reshape data to collapse samples and time steps into one dimension
    data_reshaped = data.reshape(num_samples * num_time_steps, num_features)
    
    # Create DataFrame from the reshaped data
    df = pd.DataFrame(data_reshaped, columns=column_names)
    
    # Normalize the data
    normalized_data = scaler.denormalize(df)
    
    # Sum specified columns to create 'Wert'
    summed_values = normalized_data[column_names].sum(axis=1)
    summed_array = summed_values.to_numpy()
    
    # Reshape summed array back to original data shape with one feature
    final_array = summed_array.reshape(num_samples, num_time_steps, 1)
    
    # Normalize the summed array
    final_array_2d = final_array.reshape(-1, 1)
    scaler_new.fit(final_array_2d)
    sum_normalized = scaler_new.transform(final_array_2d)
    
    # Reshape normalized data back to original shape
    sum_normalized = sum_normalized.reshape(num_samples, num_time_steps, 1)
    
    return sum_normalized