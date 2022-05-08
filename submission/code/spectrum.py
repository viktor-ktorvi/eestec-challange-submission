import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def my_fft(x_downsampled, Fs, donwsample_factor=1, Nfft=1024, window_type=None):
    if window_type is None:
        window = np.ones(len(x_downsampled))
    elif window_type == 'hamming':
        window = np.hamming(len(x_downsampled))
    else:
        raise NotImplementedError()

    fourier = np.fft.fft(x_downsampled * window, n=Nfft)
    amp = np.abs(fourier * 2 / len(x_downsampled))
    phase = np.angle(fourier)

    faxis = np.fft.fftfreq(n=Nfft) * Fs / donwsample_factor

    return amp, phase, faxis


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


if __name__ == '__main__':
    """
    0. Do we bandpass filter before hand? Yeah, why not?
    1. Downsample signal to get Fs to the 0-5 Hz (or less) range for better sampling of the spectrum
    2. Window that signal to kill the side lobes
    3. Calculate fft and abs(fft)
    4. Find peaks
    5. Sort peaks by height
    6. Take the highest peaks (2 or how many you expect)
    7. Read frequencies
    """
    Fs = 1000
    t = np.arange(start=0, stop=3, step=1 / Fs)
    f = [0.5, 1.5]
    a = [1, 4]
    x = 0
    # TODO add noise. Or just look at the real signal
    for i in range(len(f)):
        x += a[i] * np.sin(2 * np.pi * f[i] * t)

    flim = 5

    donwsample_factor = 100
    Nfft = 2 ** 14

    fs_downsampled = Fs / donwsample_factor
    # TODO it seems to be better to filter a downsampled signal. Try it out
    #  or no? maybe the visualization has bas resolution. We have to try it out

    plt.figure()
    plt.title('Different filer orders for heart rate')
    for order in [1, 2, 3, 4, 5, 6, 7, 8]:
        b, a = butter_bandpass(lowcut=0.8, highcut=2, fs=fs_downsampled, order=order)
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs_downsampled * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.xlim([0, 5])
    plt.xlabel('f Hz')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Different filer orders for breathing rate')
    for order in [1, 2, 3, 4, 5, 6]:
        b, a = butter_bandpass(lowcut=0.2, highcut=0.5, fs=fs_downsampled, order=order)
        w, h = signal.freqz(b, a, worN=2000)
        plt.plot((fs_downsampled * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.xlim([0, 5])
    plt.xlabel('f Hz')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, x)
    plt.show()

    b, a = butter_bandpass(lowcut=0.8, highcut=2, fs=Fs, order=3)
    x = signal.filtfilt(b, a, x)

    x_downsampled = x[::donwsample_factor]
    amp, phase, faxis = my_fft(x_downsampled, Fs, donwsample_factor, Nfft, window_type='hamming')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(faxis, amp)
    ax[0].set_title('Amp')
    ax[0].set_xlim(left=-flim, right=flim)
    ax[1].plot(faxis, phase)
    ax[1].set_title('Phase')
    ax[1].set_xlim(left=-flim, right=flim)

    fig.tight_layout()
    fig.show()

    right_lim = int(np.floor(Nfft / 2))
    right_side_amp = amp[:right_lim]
    peaks = signal.find_peaks(right_side_amp)

    sorted_peak_indices = np.argsort(right_side_amp[peaks[0]])[::-1]
    sorted_peaks = peaks[0][sorted_peak_indices]

    highest_peaks = sorted_peaks[:1]

    plt.figure()
    plt.plot(faxis[:right_lim], right_side_amp)
    plt.scatter(faxis[highest_peaks], right_side_amp[highest_peaks], color='r')
    plt.show()

    print('Peak frequencies: ', faxis[highest_peaks], ' Hz')
