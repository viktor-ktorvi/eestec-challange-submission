from utils import get_filenames, load_as_dataframe
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA


def get_chunks(dfs, chunk_size, subtract_mean=True):
    chunks = []
    for i in range(len(dfs)):
        chunks += np.array_split(dfs[i], len(dfs[i]) // chunk_size)

    if subtract_mean:
        for i in tqdm(range(len(chunks))):
            chunks[i] -= chunks[i].mean()

    return chunks


def extract_features(dfs, chunk_size, n_pca=None, subtract_mean=True):
    chunks = []
    for i in range(len(dfs)):
        chunks += np.array_split(dfs[i], len(dfs[i]) // chunk_size)

    num_features = 7 * chunks[0].shape[1]
    feature_matrix = np.zeros((num_features, len(chunks)))
    for i in tqdm(range(len(chunks))):
        if subtract_mean:
            chunks[i] -= chunks[i].mean()
        single_chunk = chunks[i].describe()
        transposed = single_chunk.T
        feature_matrix[:, i] = transposed.drop(
            columns=['count']).to_numpy().flatten()  # the count column is not important

    feature_matrix_df = pd.DataFrame(feature_matrix)

    if n_pca is not None:
        pca = PCA(n_components=n_pca)
        pca.fit(feature_matrix_df)

        low_dim_features_df = pd.DataFrame(pca.components_)
        return low_dim_features_df
    else:
        return feature_matrix_df


def extract_from_folder(folder, subfolder, extension, max_files, shuffle=False, chunk_size=1000, n_pca=7):
    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=shuffle)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    return extract_features(dfs, chunk_size, n_pca=n_pca)


if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 100
    chunk_size = 1000
    n_pca = 2

    features_quasi = extract_from_folder(folder, subfolder, extension, max_files, shuffle=False, chunk_size=chunk_size,
                                         n_pca=n_pca)

    labels_quasi = pd.DataFrame(np.zeros(features_quasi.shape[1]))

    subfolder = 'moving'

    features_moving = extract_from_folder(folder, subfolder, extension, max_files, shuffle=False, chunk_size=chunk_size,
                                          n_pca=n_pca)

    labels_moving = pd.DataFrame(np.zeros(features_moving.shape[1]))

    f_quasi_np = features_quasi.to_numpy()
    f_moving_np = features_moving.to_numpy()

    plt.figure()
    plt.scatter(f_quasi_np[0, :], f_quasi_np[1, :], marker='x', color='b')
    plt.scatter(f_moving_np[0, :], f_moving_np[1, :], marker='x', color='r')
    plt.show()
