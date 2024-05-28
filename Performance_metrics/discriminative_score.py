from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def calculate_discriminative_scores(processed_data, synthetic_data, attributes_idx, test_size=0.2, random_state=42):
    """
    Calculate discriminative scores for specified attributes using logistic regression.

    :param processed_data: Real data as a numpy array.
    :param synthetic_data: Synthetic data as a numpy array.
    :param attributes_idx: List of attribute indices to calculate scores for.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Seed used by the random number generator.
    :return: Dictionary containing discriminative scores for each attribute index.
    """
    discriminative_scores = {}

    for attr_idx in attributes_idx:
        # Prepare the data
        real_values = processed_data[:, attr_idx, :].flatten()  # Flattening to make it 1-dimensional
        synthetic_values = synthetic_data[:, attr_idx, :].flatten()  # Flattening to make it 1-dimensional
        X = np.concatenate((real_values, synthetic_values)).reshape(-1, 1)
        y = np.concatenate((np.ones(len(real_values)), np.zeros(len(synthetic_values))))

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train the classifier
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate the discriminative score
        score = 1 - accuracy_score(y_test, y_pred)
        discriminative_scores[attr_idx] = score

    return discriminative_scores
