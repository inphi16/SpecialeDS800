import numpy as np
from scipy.stats import pearsonr

def calculate_pearson_correlations(real_data, synthetic_data):
    pearson_corrs_synthetic = []
    # pearson_corrs_synthetic_dtw = []
    
    for feature_idx in range(real_data.shape[2]):
        real = real_data[:, :, feature_idx].flatten()
        synthetic = synthetic_data[:, :, feature_idx].flatten()
        # synthetic_dtw = synthetic_dtw_data[:, :, feature_idx].flatten()
        
        corr_synthetic, _ = pearsonr(real, synthetic)
        # corr_synthetic_dtw, _ = pearsonr(real, synthetic_dtw)
        
        pearson_corrs_synthetic.append(corr_synthetic)
        # pearson_corrs_synthetic_dtw.append(corr_synthetic_dtw)
    
    return pearson_corrs_synthetic #, pearson_corrs_synthetic_dtw
