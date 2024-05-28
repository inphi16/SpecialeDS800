from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

def build_and_train_lstm(processed_data, synthetic_data, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    # Determine sequence length from data
    sequence_length = processed_data.shape[1]  # Assuming processed_data is (n_samples, sequence_length, n_features)
    
    # Concatenate and prepare labels
    X = np.concatenate((processed_data, synthetic_data), axis=0)
    y = np.concatenate((np.ones(len(processed_data)), np.zeros(len(synthetic_data))))
    
    # Shuffle and split the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the LSTM model with two layers
    model = Sequential([
        LSTM(16, input_shape=(sequence_length, X.shape[2]), return_sequences=True),  # First LSTM layer
        LSTM(8, return_sequences=False),  # Second LSTM layer
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    print('Discriminative score:', 1 - test_acc)

    return model, 1-test_acc
