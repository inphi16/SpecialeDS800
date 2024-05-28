import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error

def calculate_metrics(y_real_test, real_predictions, synth_predictions):
    n_features = y_real_test.shape[1] if len(y_real_test.shape) > 1 else 1
    feature_names = [f'Feature_{i}' for i in range(n_features)]

    metrics_dict = {
        'Feature': feature_names,
        'R2 Score Real': [],
        'R2 Score Synth': [],
        'MAE Real': [],
        'MAE Synth': [],
        'MSLE Real': [],
        'MSLE Synth': []
    }

    for i in range(n_features):
        y_test_feature = y_real_test[:, i] if n_features > 1 else y_real_test
        real_pred_feature = real_predictions[:, i] if n_features > 1 else real_predictions
        synth_pred_feature = synth_predictions[:, i] if n_features > 1 else synth_predictions

        # Calculate R2 Score, MAE
        metrics_dict['R2 Score Real'].append(r2_score(y_test_feature, real_pred_feature))
        metrics_dict['R2 Score Synth'].append(r2_score(y_test_feature, synth_pred_feature))
        metrics_dict['MAE Real'].append(mean_absolute_error(y_test_feature, real_pred_feature))
        metrics_dict['MAE Synth'].append(mean_absolute_error(y_test_feature, synth_pred_feature))

        # Check if the feature contains zero or negative values for MSLE computation
        if np.all(y_test_feature > 0) and np.all(real_pred_feature > 0) and np.all(synth_pred_feature > 0):
            metrics_dict['MSLE Real'].append(mean_squared_log_error(y_test_feature, real_pred_feature))
            metrics_dict['MSLE Synth'].append(mean_squared_log_error(y_test_feature, synth_pred_feature))
        else:
            # Use np.nan for features that cannot have MSLE computed
            metrics_dict['MSLE Real'].append(np.nan)
            metrics_dict['MSLE Synth'].append(np.nan)
            print(f"Feature {i} contains zero or negative values, MSLE cannot be computed for this feature.")

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(metrics_dict)

    return results_df
