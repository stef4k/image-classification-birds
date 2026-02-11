import torch.nn as nn

# this is a custom head that we tried out
# it is not used in the final version as it did not improve performance (see report)
class CustomHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.5):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # (GELU matches ConvNeXt internals)
            nn.GELU(),
            # dynamic dropout
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)