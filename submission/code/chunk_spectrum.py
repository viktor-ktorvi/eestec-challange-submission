from stats import get_chunks
from utils import load_as_dataframe, get_filenames
from spectrum import my_fft
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from spectrum import butter_bandpass
from scipy import signal


def chunk_fft(x_downsampled, Fs, donwsample_factor, Nfft, window_type, columns, show=False, num_peaks=1,
              title='Frequency'):
    right_lim = int(np.floor(Nfft / 2))

    if show:
        fig_fft, ax_fft = plt.subplots(len(columns), 1)
        fig_fft.suptitle(title)

    peak_freqs = {}

    for i, column in enumerate(columns):
        amp, phase, faxis = my_fft(x_downsampled[column].to_numpy(), Fs, donwsample_factor, Nfft,
                                   window_type=window_type)
        right_side_amp = amp[:right_lim]
        right_faxis = faxis[:right_lim]

        peaks = signal.find_peaks(right_side_amp)

        sorted_peak_indices = np.argsort(right_side_amp[peaks[0]])[::-1]
        sorted_peaks = peaks[0][sorted_peak_indices]

        highest_peaks = sorted_peaks[:num_peaks]

        if show:
            ax_fft[i].set_title(column)
            ax_fft[i].plot(right_faxis, right_side_amp)
            ax_fft[i].scatter(right_faxis[highest_peaks], right_side_amp[highest_peaks], color='r')

        peak_freqs[column] = right_faxis[highest_peaks]

    if show:
        fig_fft.tight_layout()
        fig_fft.show()

    return peak_freqs


def filter_dataframe(b, a, df):
    new_dict = {}
    for column in df.columns:
        new_dict[column] = signal.filtfilt(b, a, df[column].to_numpy())

    return pd.DataFrame.from_dict(new_dict)


if __name__ == '__main__':
    folder = 'data'
    subfolder = 'activity-anxionsness'
    extension = '.npz'
    max_files = 1

    filenames = get_filenames(folder=folder, subfolder=subfolder, extension=extension, max_files=max_files,
                              shuffle=True)
    dfs = [load_as_dataframe(os.path.join(folder, subfolder), f) for f in filenames]

    chunk_size = 3000
    Fs = 1000
    donwsample_factor = 100
    Nfft = 2 ** 8

    chunks = get_chunks(dfs, chunk_size, subtract_mean=True)

    chunk = chunks[30]

    # time
    fig_time, ax_time = plt.subplots(len(chunk.columns), 1)

    fig_time.suptitle('Time')

    for i, column in enumerate(chunk.columns):
        ax_time[i].set_title(column)
        ax_time[i].plot(chunk[column].to_numpy())

    fig_time.tight_layout()

    fig_time.show()

    # Frequency
    window_type = 'hamming'
    x_downsampled = chunk[::donwsample_factor]
    peak_freqs_no_filtering = chunk_fft(x_downsampled, Fs, donwsample_factor, Nfft, window_type, columns=chunk.columns,
                                        show=True,
                                        num_peaks=2, title='No filtering')

    # TODO Downsample before filtering?

    b_heart, a_heart = butter_bandpass(lowcut=0.8, highcut=2, fs=Fs, order=3)
    b_breath, a_breath = butter_bandpass(lowcut=0.2, highcut=0.5, fs=Fs, order=3)

    heart_chunk = filter_dataframe(b_heart, a_heart, chunk)
    x_heart_downsampled = heart_chunk[::donwsample_factor]
    peak_freqs_heart = chunk_fft(x_heart_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                 columns=heart_chunk.columns,
                                 show=True,
                                 num_peaks=1, title='Heart')

    breath_chunk = filter_dataframe(b_breath, a_breath, chunk)
    x_breath_downsampled = breath_chunk[::donwsample_factor]
    peak_freqs_breath = chunk_fft(x_breath_downsampled, Fs, donwsample_factor, Nfft, window_type,
                                  columns=heart_chunk.columns,
                                  show=True,
                                  num_peaks=1, title='Breath')
