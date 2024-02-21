from src.data import * 

dataset = CubDataset()

with open(os.path.join("data/CUB_200_2011", "classes.txt"), "r") as f: 
    print([elem.split(".")[1] for elem in f.read().splitlines()])






