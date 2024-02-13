import torch
import torch.nn as nn 
import os 
from torchvision import datasets 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

device = (
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
