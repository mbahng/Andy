
import pandas as pd
from torch.utils.data import Dataset
from src.utils import taxonomy_level_array
from torchvision.datasets.folder import default_loader
import os
from PIL import Image

class BioscanDataset(Dataset): 

    def __init__(self, train:bool = True): 

        self.base_path = os.path.join("data", "BIOSCAN-1M")
        self.image_path = os.path.join(self.base_path, "images", "cropped_256")
        self.taxonomy_level_array = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"] 

        if train: 
            self.data = pd.read_csv(os.path.join(self.base_path, "train.tsv"), sep='\t')
        else: 
            self.data = pd.read_csv(os.path.join(self.base_path, "test.tsv"), sep='\t')

    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, idx): 

        image = Image.open(os.path.join(self.image_path, self.data.iloc[idx]["image_file"]))
        genetics = self.data.iloc[idx]["nucraw"]
        taxonomy = [self.data.iloc[idx][c] for c in self.taxonomy_level_array]

        return image, genetics, taxonomy

class Cub2011(Dataset):
    base_folder = os.path.join("CUB_200_2011", "images")

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.loader = default_loader
        self.train = train

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        return img, target

# class CubDataset(Dataset): 
#
#     def __init__ (self, train:bool = True): 
#
#         self.base_path = os.path.join("data", "CUB_200_2011")
#
#         self.image_ids = pd.read_csv(os.path.join(self.base_path, "images.txt"), sep=' ', header=None, names=["id", "file_name"])
#
#         if train: 
#
#
#         pass 
#
#     def __len__(self): 
#         pass 
#
#     def __getitem__(self, idx): 
#         pass 

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
