import warnings

# Suppress warnings that contain 'threadpoolctl' in the message
warnings.filterwarnings("ignore", message=".*threadpoolctl.*")

# Your code that might trigger the warning
import numpy as np
import threadpoolctl
from sklearn.model_selection import train_test_split

def create_split_indices(total_samples, test_size, validation_size):
    """
    Create indices for the training, validation, and testing splits.
    """
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Shuffle the indices
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

# # Create split indices based on the total number of sequences
# total_sequences = processed_data.shape[0]
# train_indices, validation_indices, test_indices = create_split_indices(
#     total_samples=total_sequences,
#     test_size=0.10,
#     validation_size=0.15
# )

# # Assuming target_col_indices = [0, 1] because 'wert' is at index 0 and 'temp' is at index 1
# target_col_indices = [0,1]

# # Split the data using the indices
# X_real_train, X_real_validation, X_real_test, y_real_train, y_real_validation, y_real_test = split_data_with_indices(
#     processed_data=processed_data,
#     train_indices=train_indices,
#     validation_indices=validation_indices,
#     test_indices=test_indices,
#     target_col_indices=target_col_indices
# )

# X_synth_train, X_synth_validation, X_synth_test, y_synth_train, y_synth_validation, y_synth_test = split_data_with_indices(
#     processed_data=synthetic_data,
#     train_indices=train_indices,
#     validation_indices=validation_indices,
#     test_indices=test_indices,
#     target_col_indices=target_col_indices
# )