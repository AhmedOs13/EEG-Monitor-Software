# EEG Monitor Software (Python)
## Intro. to EEG
The EEG, commonly referred to as brain waves, reflects the electrical activity within the brain. EEGs are ordered to diagnose conditions that may affect the brain. During an EEG test, a technician places small metal discs, known as electrodes, on your scalp. These electrodes transmit the electrical signals generated by your brain cells to a machine, allowing monitoring and providing insights into the functioning of various brain regions.

## Setting Up Essential Libraries
In this section, the groundwork is laid by importing crucial libraries necessary for our EEG monitor software. These libraries serve as the backbone, enabling EEG data to be manipulated, analyzed, and visualized effectively.

`mne`: Essential for EEG data handling, offering robust tools for preprocessing, analysis, and visualization in neuroimaging.

`math`: Provides essential mathematical functions for EEG signal processing, aiding in basic arithmetic and advanced computations.

`numpy`: Fundamental for numerical computing, facilitating efficient processing of EEG data arrays through powerful array operations and mathematical functions.

`scipy.signal`: Offers critical signal processing functionalities for EEG data analysis, including filtering and spectral analysis.

`matplotlib.pyplot`: Enables creation of insightful visualizations for EEG data analysis, leveraging a wide range of plotting functions within the versatile Matplotlib library.

```
import mne
from math import *
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
```

## Accessing and Visualizing EEG Data
In this section, the focus is placed on the process of reading EEG data from a file and preparing it for analysis. EEG Datasets: [https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441)

The process is demonstrated through a code segment that loads EEG data from a file (`h01.edf`) and organizes it into epochs for subsequent analysis. Essential information such as the number of epochs and channels is extracted, and a sample EEG signal is visualized before undergoing any preprocessing steps.

```
edf_file_path = r"E:\JupyterNotebook\dataverse_files\h01.edf"
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
```
![image](https://github.com/user-attachments/assets/2f869b04-859e-47ce-a2bf-814ad2332e4e)


## Resampling Data
In this section, We resampling the raw signal at a different frequency. The resulting data, now containing noise, is visualized to demonstrate the impact of the added noise on the EEG signal.

```
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
```
![image](https://github.com/user-attachments/assets/e3a47e71-82b6-4bd4-bc8c-6d6aa0300118)


## Signal Filtering for Enhanced Clarity
In this section, signal filtering techniques are employed to enhance the clarity of EEG data. The displayed code snippet exemplifies the application of these techniques to resampled EEG data. Through the specification of frequency ranges and the implementation of notch filtering, noise and unwanted frequencies are effectively diminished, leading to the production of clearer and more interpretable EEG signals. The visual representation underscores the significant role of signal filtering in improving the quality of EEG data for subsequent analysis.

```
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
```
![image](https://github.com/user-attachments/assets/e4501459-e9ff-4f80-bc62-97904be74ddc)


## Exploring Time-Frequency Characteristics
This section explores the time-frequency characteristics of EEG signals. The provided code snippet calculates the average frequencies across epochs for each EEG channel. This frequency information is obtained by analyzing the time intervals between signal peaks. The resulting frequency matrix offers insights into the dynamic time-frequency characteristics of EEG signals, enabling further exploration and interpretation of brain activity patterns.

```
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
```
![image](https://github.com/user-attachments/assets/b84cca3a-c010-46b2-854a-5d5a8a4f2073)


## Exploring Power Spectrum Density (PSD)
In this section, the Power Spectrum Density (PSD) of EEG signals is explored to comprehend their power characteristics and frequency distribution. The showcased code snippet produces a plot illustrating the PSD of the filtered EEG signals. Through the examination of power distribution across different frequency bands, the patterns of underlying neural activity can be inferred. This analysis constitutes a crucial step towards understanding the spectral properties of EEG signals, aiding in the identification of significant frequency components and their impact on brain dynamics.

```
raw_filtered.plot_psd()
```
![image](https://github.com/user-attachments/assets/2a3f265a-ac9b-49ed-84cb-cbc453446146)


## EEG Final Signal Plot
In this final plot, the amplitude variation over time for three selected EEG channels is presented. Each subplot is a representation of a distinct EEG channel, with time in seconds denoted by the x-axis and amplitude in microvolts (uV) indicated by the y-axis. This visualization provides valuable insights into the temporal dynamics of EEG signals, assisting in the interpretation of neural activity patterns.

```
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
```
![image](https://github.com/user-attachments/assets/16d36ac6-626d-461e-ad09-c123c1f46204)


