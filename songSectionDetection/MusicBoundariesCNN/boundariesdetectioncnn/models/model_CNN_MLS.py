import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1(nn.Module):
    def __init__(self, output_channels):
        super(CNN_1, self).__init__()
        self.output_channels = output_channels
        
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.output_channels, 
            kernel_size=(5, 7), 
            stride=(1, 1),
            padding=((5 - 1) // 2, (7 - 1) // 2)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 3), stride=(5, 1), padding=(1, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)
        
        return x
    
class CNN_2(nn.Module):
    def __init__(self, input_channels):
        super(CNN_2, self).__init__()
        self.input_channels = input_channels

        self.conv2 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.input_channels // 2,
            kernel_size=(3, 5),
            stride=(1, 1),
            padding=((3 - 1) // 2, (5 - 1) * 3 // 2),
            dilation=(1, 3)
        )

        self.dropout1 = nn.Dropout2d()
        self.lineal1 = nn.Conv1d(self.input_channels // 2, 1024, 1)

        self.dropout2 = nn.Dropout1d(0.4)
        self.lineal2 = nn.Conv1d(1024, 1, 1)

    def forward(self, x):
        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))  # Flatten the tensor

        x = self.dropout1(x)
        x = self.lineal1(x)
        x = F.leaky_relu(x)

        x = x.view(-1, 1024, 1, 1)  # Reshape the tensor

        x = self.dropout2(x)
        x = self.lineal2(x)

        return x


class CNN_Fusion(nn.Module):
    def __init__(self, output_channels1, input_channels2):
        super(CNN_Fusion, self).__init__()
        self.output_channels1 = output_channels1
        self.input_channels2 = input_channels2

        self.cnn1 = CNN_1(self.output_channels1)
        self.cnn2 = CNN_2(self.input_channels2)
    
    def forward(self, sslm):
        cnn1_out = self.cnn1(sslm)
        cnn2_out = self.cnn2(cnn1_out)
        return cnn2_out

## ...

# Load the pre-trained model
model = CNN_Fusion(output_channels1=32, input_channels2=64)

# Specify the path to the saved state dictionary
model_path = r"C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\github\FretLabsUtil\songSectionDetection\MusicBoundariesCNN\pretrained_weights\mel\saved_model_180epochs.bin"

# Load the state dictionary into memory
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# ...

# ...

# Create a new state dictionary with matching keys
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('cnn1.conv1.'):
        new_state_dict[key] = value
    elif key.startswith('cnn2.conv2.'):
        weight = value[:, :32, :, :]
        print(f"Original weight shape: {value.shape}")
        print(f"Weight shape after indexing: {weight.shape}")
        new_state_dict[key] = torch.zeros((32, 32, 3, 5))
        new_state_dict[key][:weight.shape[0], :, :weight.shape[2], :weight.shape[3]] = weight.permute(1, 0, 3, 2)[:weight.shape[0], :, :, :]
    elif key.startswith('cnn2.conv2.bias'):
        new_state_dict[key] = value[:32]
    elif key.startswith('cnn2.lineal1.'):
        weight = value[:1024, :32]
        print(f"Original weight shape: {value.shape}")
        print(f"Weight shape after indexing: {weight.shape}")
        new_state_dict[key] = torch.zeros((1024, weight.shape[0], weight.shape[1]))
        new_state_dict[key][:weight.shape[0], :, :] = weight
    elif key.startswith('cnn2.lineal1.bias'):
        new_state_dict[key] = value[:1024]
    elif key.startswith('cnn2.lineal2.'):
        weight = value[:1, :32, :, :]
        print(f"Original weight shape: {value.shape}")
        print(f"Weight shape after indexing: {weight.shape}")
        new_state_dict[key] = torch.zeros((1, 1024, 1, 1))
        new_state_dict[key][:, :weight.shape[1], :, :] = weight.permute(1, 0, 3, 2)

model.load_state_dict(new_state_dict)
model.eval()
