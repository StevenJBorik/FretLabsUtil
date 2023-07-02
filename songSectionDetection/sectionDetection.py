import numpy as np
import librosa
from scipy.signal import find_peaks
import math

song_file = "C:/Dev/git/testbuildw/songSectionDetection/rhcp.mp3"
song_data, _ = librosa.load(song_file)

spectral_flux = np.sqrt(np.sum(np.diff(np.abs(librosa.stft(song_data)))**2, axis=0))

peaks, _ = find_peaks(spectral_flux, distance=100)

sampling_rate = librosa.get_samplerate(song_file)
duration = librosa.get_duration(filename=song_file)

timestamps = peaks * duration / len(song_data)

minutes = np.floor(timestamps / 60)
seconds = np.round(timestamps % 60, decimals=2)

formatted_timestamps = ["{:02d}:{:05.2f}".format(int(m), s) for m, s in zip(minutes, seconds)]

print("Change Points: ", peaks)
print("Timestamps: ", formatted_timestamps)
