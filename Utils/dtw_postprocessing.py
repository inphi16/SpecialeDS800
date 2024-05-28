# pip install dtaidistance
import numpy as np
from dtaidistance import dtw

# Function to calculate DTW distance and adjust the sequence using the DTW path for single feature sequences
def adjust_sequence_single_feature(feature_a, feature_b):
    # Ensure that the features are of type np.float64
    feature_a = np.asarray(feature_a, dtype=np.float64).flatten()
    feature_b = np.asarray(feature_b, dtype=np.float64).flatten()
    distance = dtw.distance_fast(feature_a, feature_b)
    path = dtw.warping_path_fast(feature_a, feature_b)
    adjusted_sequence = np.zeros_like(feature_a)
    for seq_index, synth_index in path:
        adjusted_sequence[seq_index] = feature_b[synth_index]
    return adjusted_sequence

# Function to adjust all sequences in multi-feature datasets
def adjust_all_sequences(processed_data, synthetic_data):
    all_adjusted_series = []
    for index in range(len(processed_data)):
        adjusted_series = []
        for feature_idx in range(processed_data.shape[2]):  # Assuming 3rd dimension is features
            feature_a = processed_data[index, :, feature_idx]
            feature_b = synthetic_data[index, :, feature_idx]
            adjusted_feature_series = adjust_sequence_single_feature(feature_a, feature_b)
            adjusted_series.append(adjusted_feature_series)
        all_adjusted_series.append(np.stack(adjusted_series, axis=-1))
    return np.array(all_adjusted_series)


# synthetic_dtw_data = adjust_all_sequences(processed_data, synthetic_data)

def adjust_sequence_single_feature(feature_a, feature_b):
    feature_a = np.asarray(feature_a, dtype=np.float64).flatten()
    feature_b = np.asarray(feature_b, dtype=np.float64).flatten()

    best_distance = np.inf
    best_adjusted_sequence = None

    if len(feature_b) < len(feature_a):
        print("Error: feature_b is shorter than feature_a.")
        return np.zeros_like(feature_a)

    for start in range(len(feature_b) - len(feature_a) + 1):
        end = start + len(feature_a)
        feature_b_slice = feature_b[start:end]

        distance = dtw.distance_fast(feature_a, feature_b_slice)
        path = dtw.warping_path_fast(feature_a, feature_b_slice)

        if distance < best_distance:
            best_distance = distance
            adjusted_sequence = [feature_b_slice[synth_index] for seq_index, synth_index in path]

            if len(adjusted_sequence) != len(feature_a):
                adjusted_sequence = adjusted_sequence[:len(feature_a)] + [0] * (len(feature_a) - len(adjusted_sequence))

            best_adjusted_sequence = np.array(adjusted_sequence)

        # print(f"Processed slice {start+1}/{len(feature_b) - len(feature_a) + 1}, Distance: {distance}")

    if best_adjusted_sequence is None:
        print("No valid adjusted sequence found, returning zeros.")
        return np.zeros_like(feature_a)

    return best_adjusted_sequence

def adjust_all_sequences(processed_data, synthetic_data):
    all_adjusted_series = []
    total_sequences = len(processed_data)

    for index in range(total_sequences):
        # print(f"\nProcessing sequence {index+1}/{total_sequences}")
        adjusted_series = []
        
        for feature_idx in range(processed_data.shape[2]):
            # print(f" - Adjusting feature {feature_idx+1}/{processed_data.shape[2]}")
            feature_a = processed_data[index, :, feature_idx]
            feature_b = synthetic_data[index, :, feature_idx]
            adjusted_feature_series = adjust_sequence_single_feature(feature_a, feature_b)
            adjusted_series.append(adjusted_feature_series)

        all_adjusted_series.append(np.stack(adjusted_series, axis=-1))

    return np.array(all_adjusted_series)

# Usage
# synthetic_data_large = np.asarray(time_gan.sample(len(processed_data) * 2))
# dtw_timeseries_adjusted = adjust_all_sequences(processed_data, synthetic_data_large)

