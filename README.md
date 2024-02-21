# Best Computer Vision Group 


First make sure your conda environment is set up correctly. 

Check again to make sure that you have the following. 
```
torch==2.1.0
transformers=4.35.2
open_clip_torch==2.23.0
gdown
```

Prepare datasets by first creating a data folder in the root directory. 

``` 
mkdir data && cd data

# Download the cropped 256 imgages from the BIOSCAN database 
gdown https://drive.google.com/uc?id=1Pnf_ou7TyqgZgQLANzzNpCIl7q4gvXe1

# Download the data from google drive folder, this is preprocessed by us so it is important 
gdown https://drive.google.com/drive/folders/1Oz2hd_dCrCWAutwzO4AUfE6T9Cx0z6ix -O data --folder 

# Download the CUB_200_2011 dataset

wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# Extract the files to make the directory structure look like such: 
data/
├── CUB_200_2011/
│   ├── attributes/
│   ├── images/
│   ├── parts/
│   ├── attributes.txt
│   ├── bounding_boxes.txt
│   ├── classes.txt
│   ├── image_class_labels.txt
│   ├── images.txt
│   ├── README
│   └── train_test_split.txt
├── BIOSCAN-1M/
│   ├── images/
│   ├── metadata_cleaned_columns.tsv
│   ├── metadata_cleaned_permissive.tsv
│   ├── metadata_cleaned_restrictive.tsv
│   ├── test.tsv
│   ├── train.tsv
```



