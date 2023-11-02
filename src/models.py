from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.input_size = 784
        self.hidden_sizes = [100]
        self.output_size = 10

        layers = []
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class MLR(nn.Module):
    def __init__(self, args):
        input_size = 784
        output_size = 10
        super(MLR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class SVM(nn.Module):
    def __init__(self):
        input_size = 60
        output_size = 10
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(3136, 64)
        self.lin2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x