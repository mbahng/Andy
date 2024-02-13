import torch.nn as nn
import torch
import torch.nn.functional as F

def make_cnn(): 
    pass

class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class GeneticCNN1D(nn.Module):
    """Takes a (4, length) tensor and returns a (class_count,) tensor.

    Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
    """

    def __init__(self, length:int, class_count:int):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (length // 8), 512)
        self.fc2 = nn.Linear(512, class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)






