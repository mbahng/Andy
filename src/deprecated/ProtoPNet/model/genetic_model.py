import torch.nn as nn
import torch
import torch.nn.functional as F
from src.ProtoPNet.model import PPNet

class GeneticCNN(nn.Module):
    """Takes a (4, length) tensor and returns a (class_count,) tensor.

    Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
    """

    def __init__(self, length:int, class_count:int, two_dimensional:bool=False):
        super().__init__()

        if two_dimensional:
            self.conv1 = nn.Conv2d(4, 32, kernel_size=(1,3), padding=(0,1))
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
            self.conv3 = nn.Conv2d(64, 128, kernel_size=(1,3), padding=(0,1))
            self.pool =  nn.MaxPool2d(kernel_size=(1,2), stride=2)
        else:
            self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(128 * (length // 8), 512)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        return x
    
def construct_genetic_ppnet(length:int=720, num_classes:int=10, prototype_shape:int=(600, 24, 1, 1), model_path:str=None,prototype_activation_function='log'):
    m = GeneticCNN(length, num_classes, two_dimensional=True)

    # Remove the fully connected layer
    weights = torch.load(model_path)
    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    m.load_state_dict(weights)

    return PPNet(m, (4, length), prototype_shape, None, num_classes, False, prototype_activation_function, None, True)