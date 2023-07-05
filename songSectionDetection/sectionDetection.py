import numpy as np
from pydub import AudioSegment
import ruptures as rpt

# Load the audio file
audio_path = r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\github\FretLabsUtil\songSectionDetection\nir.mp3'
audio = AudioSegment.from_file(audio_path)

# Convert audio to mono and extract the raw data
audio_mono = audio.set_channels(1)
data = np.array(audio_mono.get_array_of_samples())

# Set the desired time window for each section in seconds
window_size = 20  # Adjust this value based on your preference

# Convert window size from seconds to number of samples
window_size_samples = int(window_size * audio.frame_rate)

# Perform segmentation using Ruptures
model = "rbf"  # Change the model based on your preference
algo = rpt.Pelt(model=model).fit(data)
result = algo.predict(pen=10)

# Print the sections of the song
for i in range(len(result)):
    start_time = result[i] * window_size
    end_time = (result[i] + 1) * window_size
    print(f"Section: {start_time:.2f} - {end_time:.2f} seconds")

# from __future__ import print_function
# import msaf
# import math
# import msaf
# import yaml

# # Select audio file
# audio_file = "C:/Dev/git/testbuildw/songSectionDetection/nir.mp3"

# # Load the default configuration file
# config_file = msaf.config.get_default_config_file()

# # Modify the hop size parameter in the configuration file
# with open(config_file, 'r') as f:
#     config = yaml.safe_load(f)
# config['ffeat']['hop_size'] = 512  # Adjust the hop size

# # Save the modified configuration file
# new_config_file = "my_config.yaml"
# with open(new_config_file, 'w') as f:
#     yaml.dump(config, f)

# # Segment the file using the modified configuration file
# boundaries, labels = msaf.process(audio_file, config_file=new_config_file)
# print('Estimated boundaries:', boundaries)

# # Convert boundaries to intervals and print the section boundaries
# sections = msaf.utils.boundaries_to_intervals(boundaries)
# for i, section in enumerate(sections):
#     start_timestamp = msaf.utils.time_to_str(section[0])
#     end_timestamp = msaf.utils.time_to_str(section[1])
#     print(f"Section {i+1}: {start_timestamp} - {end_timestamp}")

    
# import librosa
# import numpy as np

# def detect_song_sections(mp3_file_path):
#     # Load the audio file
#     audio_data, sr = librosa.load(mp3_file_path)

#     # Convert audio to mono
#     audio_data_mono = librosa.to_mono(audio_data)

#     # Compute the onset strength envelope
#     onset_env = librosa.onset.onset_strength(y=audio_data_mono, sr=sr)

#     # Set the hop length based on the frame rate
#     hop_length = int(sr / 100)

#     # Find the tempo and beat frames
#     tempo, beat_frames = librosa.beat.beat_track(y=audio_data_mono, sr=sr, hop_length=hop_length)

#     # Calculate the section starts based on onset strength
#     section_starts = librosa.onset.onset_detect(onset_envelope=onset_env, backtrack=False)

#     # Calculate the section end frames
#     section_ends = np.append(section_starts[1:], len(onset_env))

#     # Convert frame indices to timestamps
#     section_start_times = librosa.frames_to_time(section_starts, sr=sr)
#     section_end_times = librosa.frames_to_time(section_ends, sr=sr)

#     # Print the section timestamps
#     for i, (section_start, section_end) in enumerate(zip(section_start_times, section_end_times)):
#         print(f"Section {i+1}: {section_start:.2f} - {section_end:.2f}")

#     # Print the tempo
#     print(f"Tempo: {tempo:.2f} BPM")

# detect_song_sections(mp3_file_path)



#----- K MEANS ------ #

# import numpy as np
# import librosa
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import ruptures as rpt


# def audio_segmentation(mp3_file):
#     # Load the audio file
#     audio_data, sr = librosa.load(mp3_file)

#     # Extract audio features
#     chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
#     mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)
#     features = np.concatenate([chroma_stft, mfcc], axis=0)

#     # Standardize the features
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features.T)

#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=2)
#     features_pca = pca.fit_transform(features_scaled)

#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=2, random_state=0)
#     labels = kmeans.fit_predict(features_pca)

#     # Perform change point detection using Pelt algorithm
#     model = rpt.Pelt().fit(features_pca)
#     change_points = model.predict(pen=1200)  # Adjust the pen value to change the threshold criteria

#     # Convert change point indices to timestamps
#     timestamps = librosa.frames_to_time(change_points, sr=sr)

#     # Print the song sections with their timestamps
#     for i in range(len(timestamps) - 1):
#         start_time = timestamps[i]
#         end_time = timestamps[i + 1]
#         start_minutes = int(start_time // 60)
#         start_seconds = int(start_time % 60)
#         end_minutes = int(end_time // 60)
#         end_seconds = int(end_time % 60)
#         print(f"Section {i+1}: {start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}")


# mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/nir.mp3"

# # Specify the path to your MP3 file
# # mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/rhcp.mp3"

# # Perform audio segmentation
# audio_segmentation(mp3_file_path)



#Specify the path to your MP3 file

# ----- DBSCAN ---- #
# import librosa

# def audio_segmentation(mp3_file_path):
#     # Load audio file
#     audio_data, sr = librosa.load(mp3_file_path)

#     # Downsample the audio to a lower sampling rate
#     target_sr = 22050  # Desired sampling rate
#     audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

#     # Compute the recurrence matrix
#     recurrence_matrix = librosa.segment.recurrence_matrix(audio_data)

#     # Set the minimum segment duration (in seconds)
#     min_segment_duration = 3

#     # Perform segmentation based on the recurrence matrix
#     sections = librosa.segment.agglomerative(recurrence_matrix, k=10)

#     # Convert section indices to timestamps
#     timestamps = librosa.frames_to_time(sections, sr=target_sr)

#     # Filter out short sections
#     timestamps_filtered = [timestamps[i] for i in range(len(timestamps) - 1) if (timestamps[i + 1] - timestamps[i]) >= min_segment_duration]
#     timestamps_filtered.append(timestamps[-1])

#     # Print the sections
#     print("Song Sections:")
#     for i in range(len(timestamps_filtered) - 1):
#         section_start = timestamps_filtered[i]
#         section_end = timestamps_filtered[i + 1]
#         print(f"Section {i + 1}: {section_start:.2f}s - {section_end:.2f}s")

# # Specify the path to your MP3 file
# mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/nir.mp3"

# # Perform audio segmentation based on repeated patterns
# audio_segmentation(mp3_file_path)

