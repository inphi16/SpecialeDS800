import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  


def plot_predictions_with_real_and_synthetic(df, X_test, y_test, real_predictions, synthetic_predictions, scalers, window_size=168):
    """
    Plot training data, test data, and two sets of predictions: real and synthetic.
    
    Parameters:
    - df: The original DataFrame with the 'Zeitpunkt' and 'Wert' columns.
    - X_test: The test set features.
    - y_test: The actual test set target values.
    - real_predictions: The predictions made using the real data model.
    - synthetic_predictions: The predictions made using the synthetic data model.
    - scalers: A dictionary of fitted MinMaxScaler objects for each feature.
    - window_size: The size of the rolling window used in the data preparation.
    """
    
    # Inverse transform the predictions if they are scaled
    wert_scaler = scalers['Wert'] if 'Wert' in scalers else None
    if wert_scaler:
        real_predictions = wert_scaler.inverse_transform(real_predictions.reshape(-1, 1))
        synthetic_predictions = wert_scaler.inverse_transform(synthetic_predictions.reshape(-1, 1))

    # Determine the split index for the training and test sets
    split_idx = len(df) - len(X_test)

    # Prepare the test set dates and actual values for plotting
    test_dates = df['Zeitpunkt'][split_idx:]
    
    # Ensure the lengths of test_dates, predictions match
    min_length = min(len(test_dates), len(real_predictions), len(synthetic_predictions))
    test_dates = test_dates[:min_length]
    real_predictions = real_predictions[:min_length].flatten()
    synthetic_predictions = synthetic_predictions[:min_length].flatten()

    # Create the plot
    plt.figure(figsize=(25, 10))
    
    # Plot training data
    plt.plot(df['Zeitpunkt'][:split_idx], df['Wert'][:split_idx], label='Training Data', color='blue')
    
    # Plot test data
    plt.plot(test_dates, df['Wert'][split_idx:split_idx + min_length], label='Test Data', color='green')
    
    # Plot real model predictions aligned with the test data dates
    plt.plot(test_dates, real_predictions, label='Real Predictions', color='orange')
    
    # Plot synthetic model predictions aligned with the test data dates
    plt.plot(test_dates, synthetic_predictions, label='Synthetic Predictions', color='red')

    # Configure the plot with titles and labels
    plt.title('Training, Test Data, Real and Synthetic Predictions')
    plt.xlabel('Date')
    plt.ylabel('Wert')
    plt.legend()
    
    # Show the plot
    plt.show()

# plot_predictions_with_real_and_synthetic(df_2023, X_test, y_test, real_predictions, synthetic_predictions, scalers, window_size=168)
