
import pandas as pd
from torch.utils.data import Dataset

class GeneticDataset(Dataset):
    """
    A dataset class for the BIOSCAN genetic data. Samples are unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

    Args:
        source (str): The path to the dataset file (csv or tsv).
        sep (str): The separator used in the dataset file. Default is "\t".
        transform (callable, optional): Optional transforms to be applied to the genetic data. Default is None.
        
    Returns:
        (genetics, label): A tuple containing the genetic data and the label (phylum, class, order, family, subfamily, tribe, genus, species, subspecies)
    """

    def __init__(self,
                 source: str,
                 sep: str = "\t",
                 transform=None
        ):
        try:
            self.data = pd.read_csv(source, sep=sep)
        except FileNotFoundError:
            print(f"File {source} not found.")
            return
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genetics = row["nucraw"]
        label = [row[c] for c in ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]]

        if self.transform:
            genetics = self.transform(genetics)

        return genetics, label