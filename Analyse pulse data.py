import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal

## DATA ANALASYS SETTINGS ##
show_original_time_plots = True
show_fft_plot = False
should_bandpass = True
show_pulse_per_minute_plot = True
calculate_SNR = True
num_samples_remove_from_start = 100
use_color = "blue"
zero_pad_len = 2**16


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

# Get data from txt file
file = open("målinger/R111.txt", "r")
content = file.readlines()
red_values = []
green_values = []
blue_values = []
fps = 30

# Convert from string in document to data
for string in content:
    channels = string.split(" ")

    red = channels[0]
    green = channels[1]
    blue = channels[2]

    red_values.append(float(red))
    green_values.append(float(green))
    blue_values.append(float(blue))

# Remove possible noisy start samples
red_values = red_values[num_samples_remove_from_start:]
green_values = green_values[num_samples_remove_from_start:]
blue_values = blue_values[num_samples_remove_from_start:]

# Remove DC component
# red_values = red_values - np.mean(red_values)
# green_values = green_values - np.mean(green_values)
# blue_values = blue_values - np.mean(blue_values)

# Remove linear component
red_values = scipy.signal.detrend(red_values)
green_values = scipy.signal.detrend(green_values)
blue_values = scipy.signal.detrend(blue_values)

sampling_period = 1/(fps)
times = np.multiply(range(len(green_values)), sampling_period)

# Filter not interested in pulse below 40 and above 180
# This is in frequency domain 2/3 hz and 3 hz
if should_bandpass:
    red_values = bandpass(red_values, [2/3, 3], 30)
    green_values = bandpass(green_values, [2/3, 3], 30)
    blue_values = bandpass(blue_values, [2/3, 3], 30)

file.close()
if show_original_time_plots:
    fig, ax = plt.subplots(3)
    ax[0].plot(times, red_values, color ="red")
    ax[1].plot(times, green_values, color = "green")
    ax[2].plot(times, blue_values, color = "blue")
    # Keep spaces, plt formats pathetically
    plt.ylabel("                                                Relative color value variations")
    plt.xlabel("Time [s]")
    plt.show()

values_to_use = []
match use_color:
    case "red":
        values_to_use = red_values
    case "green":
        values_to_use = green_values
    case "blue":
        values_to_use = blue_values


values_to_use = np.multiply(values_to_use, np.hanning(len(values_to_use)))

pulse_fft_non_zero_pad = abs(np.fft.fft(values_to_use))
frequency_step = 1/(sampling_period * len(pulse_fft_non_zero_pad))
frequencies_non_zero_pad = np.fft.fftfreq(len(pulse_fft_non_zero_pad), d = sampling_period)

## Zero pad and window filter

pulse_fft = abs(np.fft.fft(values_to_use, zero_pad_len))
frequency_step = 1/(sampling_period * len(pulse_fft))
frequencies = np.fft.fftfreq(len(pulse_fft), d = sampling_period)

if show_fft_plot:
    plt.plot(frequencies, pulse_fft)
    plt.title("Frequency components of %s" % use_color)
    plt.legend()
    plt.show()

# Pulse in pulse/minute
pulses = frequencies * 60
recorded_pulse_index = abs(np.argmax(pulse_fft))
recorded_pulse = pulses[recorded_pulse_index]
if show_pulse_per_minute_plot:
    plt.plot(pulses, pulse_fft)
    plt.title("Pulse contents of %s" % use_color)
    plt.xlabel("Pulse [BPM]")
    plt.ylabel("Relative magnitude")
    plt.xlim(0,360)
    plt.show()

# Calculate SNR
if calculate_SNR:
    inn = input("Give [startpulse(float) endpulse(float) startpulse(float) endpulse(float) ...] for relevant pulse in calculations of SNR \n :")
    inn = inn.split(" ")
    relevant_parts_of_signal = []
    start_indexes = []
    end_indexes = []

    for i in range(0, (len(inn) - 1), 2):
        start_snr_calc = float(inn[i])
        end_snr_calc = float(inn[i+1]) 

        start_index = np.argmin(np.abs(pulses - start_snr_calc))
        end_index = np.argmin(np.abs(pulses - end_snr_calc))

        start_indexes.append(start_index)
        end_indexes.append(end_index)

        relevant_parts_of_signal.append(pulse_fft[start_index:(end_index+1)])

if calculate_SNR:

    plt.plot(pulses, pulse_fft, color = "red", label = "Noise")
    for i in range(len(relevant_parts_of_signal)):
        plt.plot(pulses[start_indexes[i]:end_indexes[i]], pulse_fft[start_indexes[i]:end_indexes[i]], color = "blue", label = "Signal")
    plt.xlabel("Pulse[BPM]")
    plt.ylabel("Relvative magnitude")
    plt.legend()
    plt.xlim(0, 360)
    plt.show()

    # Only look at positive part
    pulse_fft = np.multiply(pulse_fft, pulse_fft)
    energy = np.sum(pulse_fft)

    relevant_signal_energy = 0
    for part in relevant_parts_of_signal:
        part = np.multiply(part, part)
        relevant_signal_energy += (np.sum(part) * 2) #multiply with 2 for both negative and positive part of spectrum

    noise = energy - relevant_signal_energy

    snr = relevant_signal_energy / noise
    snr_db = 10*np.log10(snr)

print("Recorded pulse was " + str(recorded_pulse))

if calculate_SNR:
    print("Snr for " + use_color + " chanel was " + str(snr) + ". in [dB] " + str(snr_db))
