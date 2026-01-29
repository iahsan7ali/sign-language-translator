import torch
import torch.nn as nn

class SignLanguageFinal(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageFinal, self).__init__()
        # 225 = 33(pose)*3 + 21(lh)*3 + 21(rh)*3
        self.lstm = nn.LSTM(225, 128, batch_first=True, num_layers=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 30_frames, 225_features)
        out, _ = self.lstm(x)
        # We only care about the output of the last frame in the sequence
        return self.fc(out[:, -1, :])