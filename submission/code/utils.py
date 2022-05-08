import os
import numpy as np
import random

import pandas as pd


def radar2dict(radar_files):
    radar_files_dict = {}
    for r_file in radar_files.files:
        radar_files_dict[r_file] = radar_files[r_file]

    return radar_files_dict


def load_radar_dict(folder, filename):
    path = os.path.join(folder, filename)
    radar_files = np.load(path)

    return radar2dict(radar_files)


def load_as_dataframe(folder, filename):
    return pd.DataFrame.from_dict(dict2pandas_compatible_dict(load_radar_dict(folder, filename)))


def dict2pandas_compatible_dict(radar_dict):
    pd_comptible_dict = {}
    abses = radar_dict['abses']
    phases = radar_dict['phases']

    for i in range(abses.shape[0]):
        pd_comptible_dict['abses' + str(i)] = abses[i, :]

    pd_comptible_dict['phases'] = phases

    return pd_comptible_dict


def get_all_files(folder, extension):
    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(file)

    return files


def get_filenames(folder, subfolder, extension, max_files, shuffle=True):
    filenames = get_all_files(os.path.join(folder, subfolder), extension)

    if shuffle:
        random.shuffle(filenames)
    if len(filenames) > max_files:
        filenames = filenames[:max_files]

    return filenames


if __name__ == '__main__':
    folder = 'data/quasi-static'
    extension = '.npz'

    print(get_all_files(folder, extension))
