import numpy as np
from sklearn.decomposition import PCA
from utils import get_filenames, load_as_dataframe
from stats import get_chunks
import os
import pandas as pd
from spectrum import butter_bandpass
from sklearn.model_selection import train_test_split
from chunk_spectrum import filter_dataframe, chunk_fft
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import svm
from train_moving_detection import extract_from_subfolders
import json
from joblib import dump
import datetime


def extract_stress_features(dfs, chunk_size, Fs, donwsample_factor, Nfft, window_type):
    chunks = get_chunks(dfs, chunk_size, subtract_mean=True)

    b_heart, a_heart = butter_bandpass(lowcut=0.8, highcut=2, fs=Fs, order=3)
    b_breath, a_breath = butter_bandpass(lowcut=0.2, highcut=0.5, fs=Fs, order=3)

    feature_list = []
    for chunk in tqdm(chunks):
        heart_chunk = filter_dataframe(b_heart, a_heart, chunk)
        heart_chunk_downsampled = heart_chunk[::donwsample_factor]

        breath_chunk = filter_dataframe(b_breath, a_breath, chunk)
        breath_chunk_downsampled = breath_chunk[::donwsample_factor]

        h_freqs_dict = chunk_fft(heart_chunk_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                 columns=heart_chunk_downsampled.columns,
                                 show=False,
                                 num_peaks=1, title='Heart')

        b_freqs_dict = chunk_fft(breath_chunk_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                 columns=breath_chunk_downsampled.columns,
                                 show=False,
                                 num_peaks=1, title='Breath')

        feature_list.append(np.concatenate(
            (pd.DataFrame.from_dict(h_freqs_dict).to_numpy(), pd.DataFrame.from_dict(b_freqs_dict).to_numpy()), axis=1))

    feature_matrix = np.concatenate(feature_list)

    return feature_matrix


def extract_stress_from_folder(folder, subfolder, extension, max_files, shuffle, chunk_size, Fs, donwsample_factor,
                               Nfft, window_type):
    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=shuffle)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    return extract_stress_features(dfs, chunk_size, Fs, donwsample_factor, Nfft, window_type)


def extract_stress_from_subfolders(folder, subfolders, extension, max_files, shuffle, chunk_size, Fs, donwsample_factor,
                                   Nfft, window_type, label_val=0):
    matrices = []
    for subfolder in subfolders:
        matrices.append(extract_stress_from_folder(folder, subfolder, extension, max_files, shuffle, chunk_size, Fs,
                                                   donwsample_factor,
                                                   Nfft, window_type))

    features = np.concatenate(matrices)
    label = np.zeros(features.shape[0]) + label_val
    return features, label


if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 100
    shuffle = False
    chunk_size = 3000
    Fs = 1000
    donwsample_factor = 100
    Nfft = 2 ** 9
    window_type = 'hamming'
    relaxed_subfolders = ['relaxed-after-activity']
    active_subfolders = ['activity-anxionsness']

    use_freq = False

    if use_freq:
        relaxed_features, relaxed_label = extract_stress_from_subfolders(folder, relaxed_subfolders, extension,
                                                                         max_files,
                                                                         shuffle,
                                                                         chunk_size, Fs,
                                                                         donwsample_factor,
                                                                         Nfft, window_type, label_val=0)

        active_features, active_label = extract_stress_from_subfolders(folder, active_subfolders, extension, max_files,
                                                                       shuffle,
                                                                       chunk_size, Fs,
                                                                       donwsample_factor,
                                                                       Nfft, window_type, label_val=1)
    else:
        relaxed_features, relaxed_label = extract_from_subfolders(folder, relaxed_subfolders, extension, max_files,
                                                                  chunk_size,
                                                                  n_pca=None,
                                                                  label_val=0)

        active_features, active_label = extract_from_subfolders(folder, active_subfolders, extension, max_files,
                                                                chunk_size,
                                                                n_pca=None,
                                                                label_val=1)

    X = np.concatenate((relaxed_features.T, active_features.T), axis=0)
    y = np.concatenate((relaxed_label, active_label))

    n_pca = None
    if n_pca is not None:
        pca = PCA(n_components=n_pca)
        pca.fit(X.T)

        X = pca.components_.T

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = svm.SVC()
    # clf = svm.NuSVC(gamma="auto")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)

    print('Accuracy = {:.2f}'.format(acc))

    hyperparams = {'chunk_size': chunk_size, 'n_pca': n_pca}
    datetime_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    savepath = 'stress_svm_acc_{:d}_'.format(round(acc * 100)) + datetime_now
    os.mkdir(savepath, 0o666)

    with open(os.path.join(savepath, 'hyperparams.json'), 'w') as fp:
        json.dump(hyperparams, fp)

    dump(clf, os.path.join(savepath, 'model.joblib'))
