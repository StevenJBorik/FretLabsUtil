# ----- K MEANS ------ #

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
#     change_points = model.predict(pen=300)  # Adjust the pen value to change the threshold criteria

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


# # Specify the path to your MP3 file
# mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/rhcp.mp3"

# # Perform audio segmentation
# audio_segmentation(mp3_file_path)



# Specify the path to your MP3 file
# mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/nir.mp3"

# ----- DBSCAN ---- #
import librosa

def audio_segmentation(mp3_file_path):
    # Load audio file
    audio_data, sr = librosa.load(mp3_file_path)

    # Perform onset detection
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)

    # Convert onset frames to timestamps
    timestamps = librosa.frames_to_time(onset_frames, sr=sr)

    # Perform energy-based segmentation
    energy = librosa.feature.rmse(y=audio_data)
    energy_threshold = 0.02 * max(energy)
    segment_boundaries = [timestamps[0]]
    for i in range(1, len(timestamps)):
        if (energy[0, onset_frames[i-1]] > energy_threshold) and (energy[0, onset_frames[i]] <= energy_threshold):
            segment_boundaries.append(timestamps[i])

    # Print the sections
    print("Song Sections:")
    for i in range(len(segment_boundaries) - 1):
        section_start = segment_boundaries[i]
        section_end = segment_boundaries[i + 1]
        print(f"Section {i + 1}: {section_start:.2f}s - {section_end:.2f}s")

# Specify the path to your MP3 file
mp3_file_path = "C:/Dev/git/testbuildw/songSectionDetection/nir.mp3"

# Perform audio segmentation using onset detection and energy-based segmentation
audio_segmentation(mp3_file_path)
