import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import savgol_filter


def plot_and_save_spectrogram(samples, sample_rate, filename):
    plt.figure(figsize=(10, 4))
    f, t, Sxx = signal.spectrogram(samples, fs=sample_rate, window='hann', nperseg=512, noverlap=256)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Частота [Hz]')
    plt.xlabel('Время [s]')
    plt.savefig(filename)
    plt.clf()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'input', 'cover.wav')
    output_path = os.path.join(current_dir, 'output')
    os.makedirs(output_path, exist_ok=True)

    sample_rate, samples = wavfile.read(input_file)

    # Генерация и сохранение исходной спектрограммы
    plot_and_save_spectrogram(samples, sample_rate, f'{output_path}/original_spectrogram.png')

    # Применение шумопонижения и сохранение результата
    filtered_samples = savgol_filter(samples, 100, 3)
    plot_and_save_spectrogram(filtered_samples, sample_rate, f'{output_path}/filtered_spectrogram.png')

    wavfile.write(f'{output_path}/filtered_audio.wav', sample_rate, filtered_samples.astype(np.int16))


if __name__ == '__main__':
    main()
