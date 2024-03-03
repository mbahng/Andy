from typing import Any
import torch
import torch.nn.functional as F
from utils.encoding import cgr_encoding

# Genetics Transformations
class GeneticOneHot(object):
    """Map a genetic string to a one-hot encoded tensor, values being in the color channel dimension.

    Args:
        length (int): The length of the one-hot encoded tensor. Samples will be padded (with Ns) or truncated to this length.
        zero_encode_unknown (bool, optional): Whether to encode unknown characters as all zeroes. Otherwise, encode them as (1,0,0,0,0). Default is True.
        include_height_channel (bool, optional): Whether to include a height channel in the one-hot encoding. Default is False.
    """

    def __init__(self, length:int=720, zero_encode_unknown: bool=True, include_height_channel: bool=False):
        self.zero_encode_unknown = zero_encode_unknown
        self.length = length
        self.include_height_channel = include_height_channel

    def __call__(self, genetic_string: str):
        """
        Args:
            genetics (str): The genetic data to be transformed.

        Returns:
            torch.Tensor: A one-hot encoded tensor of the genetic data.
        """
        # Create a dictionary mapping nucleotides to their one-hot encoding
        nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

        # Convert string to (1, 2, 1, 4, 0, ...)
        category_tensor = torch.tensor([nucleotides[n] for n in genetic_string])
        
        # Pad and crop
        category_tensor = category_tensor[:self.length]
        category_tensor = F.pad(category_tensor, (0, self.length - len(category_tensor)), value=0)

        # One-hot encode
        onehot_tensor = F.one_hot(category_tensor, num_classes=5).permute(1, 0)
        
        # Drop the 0th channel, changing N (which is [1,0,0,0,0]) to [0,0,0,0] and making only 4 classes
        if self.zero_encode_unknown:
            onehot_tensor = onehot_tensor[1:, :]

        if self.include_height_channel:
            onehot_tensor = onehot_tensor.unsqueeze(1)

        return onehot_tensor.float()


class GeneticKmerFrequency(object):
    """Map a genetic string to a k-mer frequency tensor, with values in the color channel dimensions.
    
    Args:
        length (int): The length of the frequency encoded tensor. Samples will be padded (with 0s) or truncated to this length.
        k (int): The size of each k-mer
    """
    
    def __init__(self, length:int=720, k=4):
        self.length = length
        self.k = k
        
    def __call__(self, genetic_string:str):
        """
        Args:
            genetic_string (str): The genetic sequence to be transformed.
            
        Returns:
            torch.Tensor: A 4-by-length tensor encoding frequency
        """
        
        # Create a dictionary mapping nucleotides to their one-hot encoding
        nucleotide_index = {"A": 0, "C": 1, "G": 2, "T": 3}
        
        # Convert string to (1, 2, 1, 4, 0, ...)
        category_tensor = torch.tensor([nucleotide_index[n] for n in genetic_string])
        
        # Pad and crop
        category_tensor = category_tensor[:self.length]
        category_tensor = F.pad(category_tensor, (0, self.length - len(category_tensor)), value=0)

        # One-hot encode
        one_hot = F.one_hot(category_tensor, num_classes=4)
        
        # Sliding window sum k-mer frequencies
        freq = torch.sum(one_hot[:, :self.k], dim=0)
        freq_encoding = [list(freq).copy()]
        
        for i in range(self.k, len(one_hot[0])):
            freq += one_hot[:, i] - one_hot[:, i-self.k]
            freq_encoding.append(list(freq).copy())
            
        return torch.tensor(freq_encoding)
    
    
class GeneticCGR(object):
    """Map a genetic string to a chaos game representation, specifically a list of points in a [0,1]x[0,1] region

    Args:
        length (int): The desired length of the encoded sequence. Samples will be padded (with Ns) or truncated to this length.
    """
    
    def __init__(self, length:int=720):
        self.length = length
        
    def __call__(self, genetic_string: str):
        """
        Args:
            genetics (str): The genetic data to be transformed.

        Returns:
            torch.Tensor: An n-by-2 list of points encoding the CGR.
        """
        # Truncate (padding does nothing to CGR)
        genetic_string = genetic_string[:self.length]

        return torch.from_numpy(cgr_encoding(genetic_string))
        
        

            
        
