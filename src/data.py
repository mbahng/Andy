
import pandas as pd
from torch.utils.data import Dataset
from src.utils import taxonomy_level_array

class GeneticDataset(Dataset):
    """
    A dataset class for the BIOSCAN genetic data. Samples are unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

    Args:
        source (str): The path to the dataset file (csv or tsv).
        sep (str): The separator used in the dataset file. Default is "\t".
        transform (callable, optional): Optional transforms to be applied to the genetic data. Default is None.
        drop_level (str): If supplied, the dataset will drop all rows where the given taxonomy level is not present. Default is None.
        allowed_classes ([(level, [class])]): If supplied, the dataset will only include rows where the given taxonomy level is within the given list of classes. Default is None. Use for validation and test sets.
        
    Returns:
        (genetics, label): A tuple containing the genetic data and the label (phylum, class, order, family, subfamily, tribe, genus, species, subspecies)
    """

    def __init__(self,
                 source: str,
                 sep: str = "\t",
                 transform=None,
                 drop_level: str = None,
                 allowed_classes: list[tuple[str, list[str]]]=None,
        ):
        self.data = pd.read_csv(source, sep=sep)
        self.transform = transform

        if drop_level:
            if not drop_level in taxonomy_level_array:
                raise ValueError(f"drop_level must be one of {taxonomy_level_array}")
            self.data = self.data[self.data[drop_level] != "not_classified"]

        if allowed_classes:
            for allowed_class in allowed_classes:
                level, classes = allowed_class
                if not level in taxonomy_level_array:
                    raise ValueError(f"level must be one of {taxonomy_level_array}")
                self.data = self.data[self.data[level].isin(classes)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genetics = row["nucraw"]
        label = [row[c] for c in taxonomy_level_array]

        if self.transform:
            genetics = self.transform(genetics)

        return genetics, label
    
    def get_classes(self, class_name: str):
        """Get a tuple of the list of the unique classes in the dataset, and their sizes for a given class name, e.x. order."""
        classes = self.data[class_name].unique()
        class_sizes = self.data[class_name].value_counts()

        return list(classes), list(class_sizes[classes])