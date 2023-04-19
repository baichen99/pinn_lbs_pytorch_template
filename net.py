import torch
import torch.optim
from collections import OrderedDict
import torch.nn as nn


# mlp [3] + [2] + [4]
class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = torch.nn.Linear(3, 2)
        self.fc2 = torch.nn.Linear(2, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

# 定义Net
class Net(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active = activation

        # initial_bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1: break
            i += 1
            x = self.active(x)
        return x
