import torch
import torch.nn as nn

class SignLanguageFinal(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageFinal, self).__init__()
        # 1. Must be bidirectional=True
        # 2. Hidden size must be 256
        self.lstm = nn.LSTM(225, 256, batch_first=True, num_layers=2, bidirectional=True)
        
        # The error logs show a Sequential block for the FC layer
        # fc.0 is Linear, fc.1 is BatchNorm, fc.4 is the final Linear
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 128), # 256*2 because it's bidirectional
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out
