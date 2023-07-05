
# def test_load_models():
#     name = 'mel'
#     configs.load_model(name)
    

import numpy as np
import librosa
import torch
from boundariesdetectioncnn import configs 


# Load the trained model
model = configs.load_model('mel')

# Preprocess the audio to obtain the spectrogram representation
audio_path = 'C:/Dev/git/testbuildw/songSectionDetection/nir.mp3'  # Path to the audio file
audio, sr = librosa.load(audio_path, sr=22050)  # Load the audio file

# Apply audio preprocessing using librosa
spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

# Convert the spectrogram to a PyTorch tensor
spectrogram_tensor = torch.from_numpy(spectrogram[np.newaxis, np.newaxis, :, :]).float()

# Pass the spectrogram through the model to obtain the predicted section boundaries
with torch.no_grad():
    predicted_boundaries = model(spectrogram_tensor)
    predicted_boundaries = predicted_boundaries.squeeze().numpy()

# Postprocess the predicted boundaries (You may need to customize this part)
section_boundaries = predicted_boundaries

# Print the section boundaries
for section in section_boundaries:
    start_time = section[0]  # Start time of the section in seconds
    end_time = section[1]  # End time of the section in seconds
    print(f"Section: {start_time:.2f} - {end_time:.2f} seconds")


