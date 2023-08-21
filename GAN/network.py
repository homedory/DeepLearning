import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.g_fc1 = nn.Linear(100, 128, bias=True)
        self.g_relu = nn.ReLU(inplace=True)
        self.g_fc2 = nn.Linear(128, 784, bias=True)
        self.g_sigmoid = nn.Sigmoid()

        # Initialize weight parameters
        nn.init.xavier_uniform_(self.g_fc1.weight)
        nn.init.xavier_uniform_(self.g_fc2.weight)

    def forward(self, x):
        x = self.g_fc1(x)
        x = self.g_relu(x)
        x = self.g_fc2(x)
        x = self.g_sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d_fc1 = nn.Linear(784, 128, bias=True)
        self.d_relu = nn.ReLU(inplace=True)
        self.d_fc2 = nn.Linear(128, 1, bias=True)
        self.d_sigmoid = nn.Sigmoid()

        # Initialize weight parameters
        nn.init.xavier_uniform_(self.d_fc1.weight)
        nn.init.xavier_uniform_(self.d_fc2.weight)

    def forward(self, x):
        x = self.d_fc1(x)
        x = self.d_relu(x)
        x = self.d_fc2(x)
        x = self.d_sigmoid(x)

        return x.squeeze()