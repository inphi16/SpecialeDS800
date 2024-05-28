import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Dropout, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

def CNN_GRU_regression(timesteps, features_per_timestep, units):
    model = Sequential()

    # CNN Layers
    # Assume the input shape is (timesteps, features_per_timestep)
    model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(timesteps, features_per_timestep)))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Reshape the output to be compatible with GRU layers
    # Calculate the new number of timesteps and features after pooling layers
    new_timesteps = timesteps // (2 * 2) # Assuming pool_size=2 for both pooling layers
    new_features = 16 # This is the number of filters in the last Conv1D layer
    model.add(Reshape((new_timesteps, new_features)))

    model.add(Dropout(0.5))

    # GRU Layer - no need for Flatten since we want to process temporal sequences
    model.add(GRU(units=units, return_sequences=False))

    # Output Layer
    model.add(Dense(units=1))  

    # Compile the model
    model.compile(optimizer=Adam(), loss=MeanAbsoluteError())
    
    return model


def CNN_GRU_regression_2(timesteps, features_per_timestep, units):
    model = Sequential()

    # CNN Layers
    model.add(Conv1D(filters=4, kernel_size=2, activation='relu', input_shape=(timesteps, features_per_timestep)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Reshape for compatibility with GRU layers
    new_timesteps = timesteps // 4  # Result from pooling twice with pool_size=2
    new_features = 16  # Number of filters in the last Conv1D layer
    model.add(Reshape((new_timesteps, new_features)))

    model.add(Dropout(0.5))

    # First GRU Layer
    model.add(GRU(units=units, return_sequences=True))  # Return sequences for the next GRU layer

    # Second GRU Layer
    model.add(GRU(units=units, return_sequences=False))  # Only return the last output

    # Output Layer for regression
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(), loss=MeanAbsoluteError())

    return model


