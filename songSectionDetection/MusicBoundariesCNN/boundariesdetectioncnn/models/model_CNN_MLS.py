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
    def __init__(self, output_channels):
        super(CNN_2, self).__init__()
        self.output_channels = output_channels
        
        self.conv2 = nn.Conv2d(
            in_channels=self.output_channels,
            out_channels=self.output_channels * 2,
            kernel_size=(3, 5),
            stride=(1, 1),
            padding=((3 - 1) // 2, (5 - 1) * 3 // 2),
            dilation=(1, 3)
        )
        
        self.dropout1 = nn.Dropout2d()
        self.lineal1 = nn.Conv1d(self.output_channels * 2, 1024, 1)  # Update output channels
        
        self.dropout2 = nn.Dropout1d(0.4)
        self.lineal2 = nn.Conv1d(1024, 1, 1)  # Update input channels
        
    def forward(self, x):
        x = self.conv2(x)
        x = F.leaky_relu(x)
        
        x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3])
        
        x = self.dropout1(x)
        x = self.lineal1(x)
        x = F.leaky_relu(x)
        
        x = self.dropout2(x)
        x = self.lineal2(x)
        
        return x


class CNN_Fusion(nn.Module):
    def __init__(self, output_channels1, output_channels2):
        super(CNN_Fusion, self).__init__()
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2

        self.cnn1 = CNN_1(self.output_channels1)
        self.cnn2 = CNN_2(self.output_channels2)  # Use output_channels2 as input_channels for CNN_2
    
    def forward(self, sslm):
        cnn1_out = self.cnn1(sslm)
        cnn2_out = self.cnn2(cnn1_out)
        return cnn2_out
