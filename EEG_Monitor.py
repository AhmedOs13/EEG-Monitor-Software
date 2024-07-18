import mne
from math import *
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Accessing and Visualizing EEG Data
edf_file_path = r'write your path here'  # Example: r"E:\JupyterNotebook\dataverse_files\h01.edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)
# Slicing data into epochs with duration 3 sec.
epochs = mne.make_fixed_length_epochs(raw, duration=3, overlap=1)

eeg_dataset_epoch = epochs.get_data()
channel_names = raw.info['ch_names']
sFreq = raw.info['sfreq']  # sFreq repsents number of signals per second

print("Number of epochs:", len(eeg_dataset_epoch))  # Number of epochs: 462
print("Number of channels:", len(channel_names))  # Number of channels: 19

# Plotting Extracted data
plt.figure(figsize=(12, 5))
plt.plot(epochs.times, eeg_dataset_epoch[0][0], color='blue')
plt.title('Before Resampling')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Resampling Dataset
raw_resampled = raw.copy().resample(1000)  # Changing sFreq into 1000
epochs_resampled = mne.make_fixed_length_epochs(raw_resampled, duration=3, overlap=1)
eeg_dataset_epoch_resampled = epochs_resampled.get_data()

# Plotting data after resampling
plt.figure(figsize=(12, 5))
plt.plot(epochs_resampled.times, eeg_dataset_epoch_resampled[0][0], color='red')
plt.title('After Resampling')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Signal Filtering for Enhanced Clarity
# Filtering signals with bandpass 1.0 Hz to 50.0 Hz
raw_filtered = raw_resampled.copy().filter(l_freq=1, h_freq=50, picks='eeg', method='fir')
raw_filtered.notch_filter(freqs=60, picks='eeg', method='fir')

# Slicing data into epochs with duration 3 sec.
epochs_filtered = mne.make_fixed_length_epochs(raw_filtered, duration=3, overlap=1)
eeg_dataset_filtered = epochs_filtered.get_data()

# Plotting filtered signals
plt.figure(figsize=(12, 5))
plt.plot(epochs_filtered.times, eeg_dataset_filtered[0][0], color='black')
plt.title('After Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Exploring Time-Frequency Characteristics
channel_freq = []
for ch_index in range(len(channel_names)):
    epoch_freq = []
    for ep_index in range(len(epochs)):
        eeg_channel_data = eeg_dataset_filtered[ep_index][ch_index]
        peaks, _ = signal.find_peaks(eeg_channel_data, distance=100)
        time_diffs = np.diff(peaks) / raw.info['sfreq']
        frequencies = 1 / time_diffs
        frequencies *= 2 * np.pi
        average_frequency = np.mean(frequencies)
        epoch_freq.append(average_frequency)
    channel_freq.append(epoch_freq)

frequency = np.array(channel_freq)

print('Channels ananlyzed:', frequency.shape[0])  # Channels ananlyzed: 19
print('Epochs ananlyzed:', frequency.shape[1])  # Epochs ananlyzed: 462
print(f'Frequancy: {frequency[0][0]:.2F} Hz')  # Frequancy: 11.99 Hz

# Plotting the frequency change with epochs
first_channel_freq = frequency[0]

plt.figure(figsize=(10, 6))
plt.plot(range(len(first_channel_freq)), first_channel_freq, marker='o', linestyle='-')
plt.title('Frequency Change with Epochs in the First Channel')
plt.xlabel('Epochs')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
plt.show()


# Exploring Power Spectrum Density (PSD)
raw_filtered.plot_psd()


# EEG Final Signal Plot
num_plots = 3  # Limiting to plot 3 channels
fig = plt.figure(figsize=(8, 3 * num_plots))
for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(np.arange(eeg_dataset_filtered.shape[2]) / sFreq, eeg_dataset_filtered[0, i])
    plt.title(channel_names[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.grid(True)

plt.tight_layout()
plt.show()
