import torch
import torch.nn as nn

class CNN1D_HAR(nn.Module):
    def __init__(self, num_channels=40, num_classes=12):
        super(CNN1D_HAR, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

class LSTM_HAR(nn.Module):
    pass

class CNN_LSTM_HAR(nn.Module):
    pass