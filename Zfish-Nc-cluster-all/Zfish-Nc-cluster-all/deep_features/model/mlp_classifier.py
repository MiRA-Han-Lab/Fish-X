import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.ops import MLP


class MLPV2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.0):
        super(MLPV2, self).__init__()
        self.mlp = MLP(in_channels=in_channels, hidden_channels=hidden_channels, 
                       norm_layer=nn.modules.BatchNorm1d, dropout=dropout)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        x = self.mlp(x)
        y = self.fc(x)
        return y

