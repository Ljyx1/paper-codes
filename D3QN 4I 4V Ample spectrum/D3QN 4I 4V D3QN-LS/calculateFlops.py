import os
from replay_memory import ReplayMemory
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.optim import Adam
from torchstat import stat
import thop
import Environment_marl


class DuelingDeepQNet(nn.Module):
    def __init__(self, fc1_dims=500, fc2_dims=250, fc3_dims=120, lr=0.001):
        super(DuelingDeepQNet, self).__init__()

        """共享层"""
        self.fc1 = nn.Linear(33, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        """环境值"""
        self.state_value = nn.Linear(fc3_dims, 1)
        """动作值"""
        self.action_value = nn.Linear(fc3_dims, 16)

        self.crit = nn.MSELoss()

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        x_end = self.relu3(self.fc3(x))
        V = self.state_value(x_end)
        A = self.action_value(x_end)

        Q = V + (A - torch.mean(A, dim=1, keepdim=True))

        return Q

""" def stat(model, input_size, query_granularity=1):
    ms = ModelStat(model, input_size, query_granularity)
    ms.show_report() """
stat(DuelingDeepQNet(), (33,))

class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x

stat(CNNnet(), (1,28,28))


#
# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     print(trainable_num)
#     print(total_num)
#     return {'Total': total_num, 'Trainable': trainable_num}
#
# for name, parameters in net.named_parameters():
#     print(name, ':', parameters.size())
#     get_parameter_number(net)
