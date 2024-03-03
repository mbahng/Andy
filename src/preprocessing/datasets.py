import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing.taxonomy import taxonomy_level_array, taxonomy_level_index_map
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
from PIL import Image
from preprocessing.bioscan import preprocess_bioscan 
from preprocessing.cub import preprocess_cub
from preprocessing.transformations import GeneticOneHot

def prepare_dataset(args): 
    if args.dataset == "cub": 
        preprocess_cub() 
    elif args.dataset == "bioscan": 
        preprocess_bioscan()
    else: 
        raise Exception("Invalid Dataset")

def get_dataloaders(cfg, log): 
    train_loader, train_push_loader, test_loader = None, None, None

    log("===== Getting Relevant Dataloaders =====") 

    if cfg.DATASET.NAME == "cub":

        normalize = transforms.Normalize(
            mean=cfg.DATASET.TRANSFORM_MEAN, 
            std=cfg.DATASET.TRANSFORM_STD
        )
        # all datasets
        # train set
        train_dataset = datasets.ImageFolder(
            cfg.DATASET.TRAIN_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)

        # push set
        train_push_dataset = datasets.ImageFolder(
            cfg.DATASET.TRAIN_PUSH_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]))
        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)

        # test set
        test_dataset = datasets.ImageFolder(
            cfg.DATASET.TEST_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)

        log('training set size: {0}'.format(len(train_loader.dataset)))
        log('push set size: {0}'.format(len(train_push_loader.dataset)))
        log('test set size: {0}'.format(len(test_loader.dataset)))
        log('batch size: {0}'.format(cfg.DATASET.TRAIN_BATCH_SIZE))

    elif cfg.DATASET.NAME == "bioscan":
        t = GeneticOneHot(length=cfg.DATASET.BIOSCAN.CHOP_LENGTH, zero_encode_unknown=True, include_height_channel=True)
        d_train = GeneticDataset(os.path.abspath("data/BIOSCAN-1M/small_diptera_family-train.tsv"), transform=t, drop_level=cfg.DATASET.BIOSCAN.TAXONOMY_NAME, allowed_classes=[("order", [cfg.DATASET.BIOSCAN.ORDER_NAME])], one_label=cfg.DATASET.BIOSCAN.TAXONOMY_NAME)
        classes, sizes = d_train.get_classes(cfg.DATASET.BIOSCAN.TAXONOMY_NAME)

        # Create a test dataset, dropping all samples that don't have a family label, only allowing the order Diptera, and only allowing the families present in the training set
        d_val = GeneticDataset(os.path.abspath("data/BIOSCAN-1M/small_diptera_family-validation.tsv"), transform=t, drop_level=cfg.DATASET.BIOSCAN.TAXONOMY_NAME, allowed_classes=[("order", [cfg.DATASET.BIOSCAN.ORDER_NAME]), ("family", d_train.get_classes(cfg.DATASET.BIOSCAN.TAXONOMY_NAME)[0])], one_label=cfg.DATASET.BIOSCAN.TAXONOMY_NAME, classes=classes)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(d_train, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(d_val, batch_size=32, shuffle=True)

    log("=========================================")

    return train_loader, train_push_loader, test_loader

class BioscanDataset(Dataset): 
    '''
    A dataset for the BIOSCAN-1M dataset. The dataset is a combination of images and genetic data. The genetic data is a string of nucleotides, including base pairs A, C, G, T and an unknown character N. 
    returns (Image:PIL.JpegImagePlugin.JpegImageFile, genetics:str, label:str)
    '''

    def __init__(self, train = True, transform = None, size:str = "small", min_balance_count = 100, seed=0): 
        self.base_path = os.path.join("data", "BIOSCAN-1M")
        self.image_path = os.path.join(self.base_path, "images", "cropped_256")
        self.file_path = os.path.join(self.base_path, "BIOSCAN_Insect_Dataset_metadata.tsv")
        self.min_balance_count = min_balance_count      # the minimum number of samples for each class
        self.transform = transform
        self.train = train
        self.seed = seed

        assert size in ["small" , "medium", "large"]
        self.size = size
        self.get_dataset()


    def __len__(self): 
        return len(self.data) 

    def random_sample(self, group):
        if len(group) < self.min_balance_count:
            # print(f"{group['family'].iloc[0]} does not have enough samples")
            pass

        return group.sample(n=min(self.min_balance_count, len(group)), random_state=self.seed)

    def get_dataset(self): 
        self.data = pd.read_csv(self.file_path, sep='\t')
        self.data.drop(columns=['copyright_institution', 'photographer', 'author', 'copyright_contact', 'copyright_license', 'copyright_holder', 'processid', 'uri', 'phylum', 'class', 'subfamily', 'tribe', 'genus', 'species', 'subspecies', 'name', 'order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family'], inplace=True)
        
        if self.train: 
            self.data = self.data[self.data[f'{self.size}_insect_order'] == 'train']
        else: 
            self.data = self.data[self.data[f'{self.size}_insect_order'] == 'test']

        available_family_df = self.data['family'].value_counts()[self.data['family'].value_counts() > self.min_balance_count]
        available_family = available_family_df.index.tolist()
        available_family.remove('not_classified')
        self.data = self.data[self.data['family'].isin(available_family)]

        unrelevant_sizes = ["small", "medium", "large"]
        unrelevant_sizes.remove(self.size)
        self.data.drop(columns=[f"{size}_insect_order" for size in unrelevant_sizes], inplace=True)

        self.data = self.data.groupby('family', group_keys=False).apply(lambda x: self.random_sample(x))

    def __getitem__(self, idx): 

        image = Image.open(os.path.join(self.image_path, self.data.iloc[idx]["image_file"]))
        genetics = self.data.iloc[idx]["nucraw"]
        label = self.data.iloc[idx]["family"]

        if self.transform is not None: 
            image = self.transform(image)

        return image, genetics, label

class CubDataset(Dataset):
    '''
    returns (Image:PIL.Image.Image, label:int)
    '''

    base_folder = os.path.join("data", "CUB_200_2011", "images")

    def __init__(self, train=True, transform = None):
        self.root = os.path.expanduser("./data")
        self.loader = default_loader
        self.train = train
        self.transform = transform

        with open(os.path.join("data", "CUB_200_2011", "classes.txt"), "r") as f: 
            self.classes = [elem.split(".")[1] for elem in f.read().splitlines()] 


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

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
class GeneticDataset(Dataset):
    """
        A dataset class for the BIOSCAN genetic data. Samples are unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

        Args:
            source (str): The path to the dataset file (csv or tsv).
            sep (str): The separator used in the dataset file. Default is "\t".
            transform (callable, optional): Optional transforms to be applied to the genetic data. Default is None.
            drop_level (str): If supplied, the dataset will drop all rows where the given taxonomy level is not present. Default is None.
            allowed_classes ([(level, [class])]): If supplied, the dataset will only include rows where the given taxonomy level is within the given list of classes. Default is None. Use for validation and test sets.
            one_label (str): If supplied, the label will be the value of one_class
            classes: list[int]
            
        Returns:
            (genetics, label): A tuple containing the genetic data and the label (phylum, class, order, family, subfamily, tribe, genus, species, subspecies)
    """
    def __init__(self,
                 source: str,
                 sep: str = "\t",
                 transform=None,
                 drop_level: str = None,
                 allowed_classes: list[tuple[str, list[str]]]=None,
                 one_label: str = None,
                 classes: list[str] = None
        ):
        self.data = pd.read_csv(source, sep=sep)
        self.transform = transform
        self.one_label = one_label
        self.classes = classes

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

        if self.one_label:
            if self.classes:
                self.classes = {
                    c: i for i,c in enumerate(classes)
                }
            else:
                self.classes = {
                    c: i for i,c in enumerate(self.get_classes(self.one_label)[0])
                }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genetics = row["nucraw"]
        label = [row[c] for c in taxonomy_level_array]

        if self.transform:
            genetics = self.transform(genetics)

        if self.one_label:
            label = label[taxonomy_level_array.index(self.one_label)]
            label = torch.tensor(self.classes[label])
            
        return genetics, label
    
    def get_classes(self, class_name: str):
        """Get a tuple of the list of the unique classes in the dataset, and their sizes for a given class name, e.x. order."""
        classes = self.data[class_name].unique()
        class_sizes = self.data[class_name].value_counts()

        return list(classes), list(class_sizes[classes])
