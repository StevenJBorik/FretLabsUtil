import torch
import torchaudio
from boundariesdetectioncnn.models.model_CNN_MLS import CNN_Fusion

# Load the pre-trained model
model = CNN_Fusion(output_channels1=32, output_channels2=64)  # Update output_channels
# Specify the path to the saved state dictionary
model_path = r"C:\Dev\git\testbuildw\songSectionDetection\MusicBoundariesCNN\pretrained_weights\mel\saved_model_180epochs.bin"

# Load the state dictionary into memory
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# Modify the state dictionary to match the updated model architecture
new_state_dict = {}
for key, value in state_dict.items():
    if 'cnn1' in key:
        key = key.replace('cnn1', 'cnn1.conv1')
    elif 'cnn2' in key:
        key = key.replace('cnn2', 'cnn2.lineal1')
        if 'weight' in key:
            new_shape = (value.shape[0], value.shape[1] // value.shape[2], 1)  # Determine correct input size
            value = value.reshape(new_shape)  # Reshape weight to match new size
        elif 'bias' in key:
            value = value.squeeze()  # Remove extra dimension from bias
    new_state_dict[key] = value

model.load_state_dict(new_state_dict)
model.eval()

audio_path = r"C:\Dev\git\testbuildw\songSectionDetection\nir.mp3"

waveform, sample_rate = torchaudio.load(audio_path)
spectrogram = torchaudio.transforms.Spectrogram()(waveform)
spectrogram_tensor = torch.unsqueeze(spectrogram.mean(dim=0), 0)  # Convert to single-channel spectrogram

# Print the shapes of intermediate outputs
cnn1_out = model.cnn1(spectrogram_tensor)
cnn2_out = model.cnn2(cnn1_out)
print("Shape of CNN1 output:", cnn1_out.shape)
print("Shape of CNN2 output:", cnn2_out.shape)

# Pass the spectrogram through the model
predicted_boundaries = model(spectrogram_tensor)
print("Shape of predicted boundaries:", predicted_boundaries.shape)

# Postprocess the predicted boundaries (You may need to customize this part)
section_boundaries = predicted_boundaries

# Print the section boundaries
for section in section_boundaries:
    start_time = section[0]  # Start time of the section in seconds
    end_time = section[1]  # End time of the section in seconds
    print(f"Section: {start_time:.2f} - {end_time:.2f} seconds")
