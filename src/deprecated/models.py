import torch.nn as nn
import torch
import torch.nn.functional as F
from  model.model import PPNet 

class GeneticCNN2D(nn.Module):
    """Takes a (4, length) tensor and returns a (class_count,) tensor.

    Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
    """

    # Drop last layer and load weights into Proto Layer

    def __init__(self, length:int, class_count:int, include_connected_layer:bool):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1,3), padding=(0,1))
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,3), stride=3)

        if include_connected_layer:
            self.fc1 = nn.Linear(128 * (length // 8), class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        if hasattr(self, 'fc1'):
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            # return F.log_softmax(x, dim=1)
        return x
    
def construct_genetic_ppnet(length:int=720, num_classes:int=10, prototype_shape:int=(600, 24, 1, 1), model_path:str=None, prototype_activation_function='log', use_cosine=False):
    m = GeneticCNN2D(length, num_classes, include_connected_layer=False)

    # Remove the fully connected layer
    weights = torch.load(model_path)
    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    m.load_state_dict(weights)

    return PPNet(m, (4, 1, length), prototype_shape, None, num_classes, False, prototype_activation_function, None, True, use_cosine=use_cosine)
