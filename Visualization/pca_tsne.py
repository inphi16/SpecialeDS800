import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca_tsne_real_vs_synth(processed_data, synth_data, sequence_length, sample_size=250):
    # Random sampling
    idx = np.random.permutation(len(processed_data))[:sample_size]
    real_sample = np.asarray(processed_data)[idx]
    synthetic_sample = np.asarray(synth_data)[idx]

    # Reshape for 2D dimensionality reduction
    processed_data_reduced = real_sample.reshape(-1, sequence_length)
    synth_data_reduced = synthetic_sample.reshape(-1, sequence_length)

    # PCA and t-SNE
    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300)

    pca.fit(processed_data_reduced)
    pca_real = pd.DataFrame(pca.transform(processed_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

    data_reduced = np.concatenate((processed_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

    # Plotting
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # PCA scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('PCA results', fontsize=20, color='red', pad=10)
    ax.scatter(pca_real.iloc[:, 0], pca_real.iloc[:, 1], c='black', alpha=0.2, label='Original')
    ax.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1], c='red', alpha=0.2, label='Synthetic')
    ax.legend()

    # TSNE scatter plot
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('TSNE results', fontsize=20, color='red', pad=10)
    ax2.scatter(tsne_results.iloc[:sample_size, 0], tsne_results.iloc[:sample_size, 1], c='black', alpha=0.2, label='Original')
    ax2.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1], c='red', alpha=0.2, label='Synthetic')
    ax2.legend()

    fig.suptitle('Validating synthetic vs real data diversity and distributions', fontsize=16, color='grey')
    plt.show()

    return fig