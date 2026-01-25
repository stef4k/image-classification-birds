import torch.nn as nn

class CustomHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.7):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.layer(x)