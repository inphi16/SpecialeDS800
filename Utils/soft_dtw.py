import tensorflow as tf

def soft_dtw(y_true, y_pred, gamma=0.1):
    """
    Computes the Soft-DTW between two batches of sequences using a soft-minimum approach.
    
    Args:
    y_true: Tensor of shape [batch_size, sequence_length, num_features]
    y_pred: Tensor of shape [batch_size, sequence_length, num_features]
    gamma: Smoothing parameter for soft minimum. Smaller values make it closer to the hard DTW.
    
    Returns:
    A scalar Tensor representing the soft-DTW distance for each sequence in the batch.
    """
    # Define the squared Euclidean distance between elements
    def squared_distance(x, y):
        return tf.reduce_sum(tf.square(x - y), axis=-1)

    seq_len = tf.shape(y_true)[1]
    batch_size = tf.shape(y_true)[0]

    # Compute distances for all combinations of sequence elements
    expanded_y_true = tf.expand_dims(y_true, 2)  # shape [batch_size, seq_len, 1, num_features]
    expanded_y_pred = tf.expand_dims(y_pred, 1)  # shape [batch_size, 1, seq_len, num_features]
    distances = squared_distance(expanded_y_true, expanded_y_pred)  # shape [batch_size, seq_len, seq_len]

    # Initialize the DTW matrix with infinite values
    dtw_matrix = tf.fill([batch_size, seq_len, seq_len], tf.float32.max)
    # Set the first element to zero
    batch_indices = tf.range(batch_size)
    dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, tf.stack([batch_indices, tf.zeros_like(batch_indices), tf.zeros_like(batch_indices)], axis=1), tf.zeros_like(batch_indices, dtype=tf.float32))
    
    # Apply the recurrence with softmin
    for i in range(1, seq_len):
        for j in range(1, seq_len):
            cost = distances[:, i, j]
            r = tf.stack([dtw_matrix[:, i-1, j], dtw_matrix[:, i, j-1], dtw_matrix[:, i-1, j-1]], axis=-1)
            soft_min = -gamma * tf.reduce_logsumexp(-r / gamma, axis=-1)
            dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, tf.stack([batch_indices, tf.fill([batch_size], i), tf.fill([batch_size], j)], axis=1), cost + soft_min)

    # Extract the final value for the soft-DTW distance
    final_distances = dtw_matrix[:, -1, -1]
    return tf.reduce_mean(final_distances)  # Optionally, return mean distance across the batch

# # Example usage
# y_true = tf.random.normal([10, 100, 64])  # Batch of 10 sequences, 100 length each, 64 features
# y_pred = tf.random.normal([10, 100, 64])
# loss = soft_dtw(y_true, y_pred, gamma=0.1)
# print("Soft-DTW loss:", loss.numpy())

