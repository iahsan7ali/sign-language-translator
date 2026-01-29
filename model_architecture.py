import torch
import torch.nn as nn

class SignLanguageFinal(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageFinal, self).__init__()
        # hidden_size MUST be 128 to match the 512-sized tensors in your error
        self.lstm = nn.LSTM(225, 128, batch_first=True, num_layers=2, bidirectional=True)
        
        # The FC layer must take 128 * 2 (for bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Pull the last time step from the bidirectional output
        out = self.fc(lstm_out[:, -1, :])
        return out
