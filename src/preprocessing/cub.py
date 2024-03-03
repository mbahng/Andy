import os
import shutil
from PIL import Image
import Augmentor
import torch
import shutil

def preprocess_cub(organize=False):
    print("====== Preprocessing CUB ======")
    crop_images() 
    train_test_split()
    if organize: 
        organize_images(os.path.join("data", "CUB_200_2011", "cub200_cropped", "train_cropped"))
        organize_images(os.path.join("data", "CUB_200_2011", "cub200_cropped", "test_cropped"))
    augment_train_images()
    print("===============================")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def preprocess_cub_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def crop_images(): 
    '''
    Crops all images in the CUB_200_2011 dataset using the bounding box coordinates 
    in bounding_boxes.txt and saves the cropped images to a new folder called cropped_image.
    '''
    source_directory = os.path.join("data", "CUB_200_2011", "images") 
    destination_directory = os.path.join("data", "CUB_200_2011", "images_copy")

    if os.path.isdir(os.path.join("data", "CUB_200_2011", "cropped_image")): 
        print("Cropped images already exist")
        if os.path.isdir(destination_directory):
            shutil.rmtree(destination_directory)
            print("Redundant images_copy folder removed")
        return 

    os.makedirs(destination_directory, exist_ok=True)
    bounding_box_file = os.path.join("data", "CUB_200_2011", "bounding_boxes.txt")
    image_folder = os.path.join("data", "CUB_200_2011", "images_copy")
    output_folder = os.path.join("data", "CUB_200_2011", "cropped_image")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each folder in the source directory
    for folder_name in os.listdir(source_directory):
        folder_path = os.path.join(source_directory, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            
            # Iterate through each file in the folder and copy it to the destination folder
            for filename in os.listdir(folder_path):
                source_file = os.path.join(folder_path, filename)
                destination_file = os.path.join(destination_directory, f"{folder_name}_{filename}")
                shutil.copy2(source_file, destination_file)

    print("Image copy complete.")

    images_file = os.path.join("data", "CUB_200_2011", "images.txt")
    image_dict = {}

    with open(images_file, "r") as file:
        for line in file:
            # Split the line into id and image name
            id, image_name = line.strip().split()

            # Replace '/' in the image name with '_'
            image_name = image_name.replace('/', '_')

            # Store in the dictionary
            image_dict[image_name] = id

    bounding_boxes_dict = {}
    with open(bounding_box_file, "r") as file:
        for line in file:
            # Split the line into id and bounding box coordinates
            id, *bbox_coordinates = line.strip().split()

            # Convert bounding box coordinates to tuple of floats
            bounding_box = tuple(map(float, bbox_coordinates))

            # Store in the dictionary
            bounding_boxes_dict[id] = bounding_box

    counter = 0
    for image_name in os.listdir(image_folder):
        # Get the id using the image_dict dictionary
        
        id = image_dict.get(image_name)

        # Check if id exists in bounding_boxes_dict
        if id in bounding_boxes_dict:
            
            counter += 1
            
            # Get the bounding box coordinates
            bounding_box = bounding_boxes_dict[id]
            
            x, y, w, h = bounding_box

            # Open the image
            image_path = os.path.join(image_folder, image_name)
            img = Image.open(image_path)

            # Crop the image using bounding box coordinates
            cropped_img = img.crop((x, y, x+w, y+h))

            # Save the cropped image to the output folder
            output_path = os.path.join(output_folder, image_name)
            cropped_img.save(output_path)

    shutil.rmtree(destination_directory)

    print(f"{counter} images cropped")

def train_test_split(): 
    images_file = os.path.join("data", "CUB_200_2011", "images.txt")
    split_file = os.path.join("data", "CUB_200_2011", "train_test_split.txt")
    source_folder = os.path.join("data", "CUB_200_2011", "cropped_image")
    train_dest = os.path.join("data", "CUB_200_2011", "cub200_cropped", "train_cropped")
    test_dest = os.path.join("data", "CUB_200_2011", "cub200_cropped", "test_cropped")

    if os.path.isdir(train_dest) and os.path.isdir(test_dest):
        print("Train and test sets already exist")
        return

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    image_dict = {}  # image to id dict
    with open(images_file, "r") as file:
        for line in file:
            # Split the line into id and image name
            id, image_name = line.strip().split()

            # Replace '/' in the image name with '_'
            image_name = image_name.replace('/', '_')

            # Store in the dictionary
            image_dict[image_name] = id

    train_test_dict = {}
    with open(split_file, "r") as file:
        for line in file:
            # Split the line into id and image name
            id, training = line.strip().split()

            # Store in the dictionary
            train_test_dict[id] = training

    training_size = 0
    testing_size = 0
    for image_name in os.listdir(source_folder):
        
        if image_name not in image_dict:
            continue
        
        id = image_dict.get(image_name)
        
        if id not in train_test_dict:
            continue
        
        train = train_test_dict.get(id)
        
        image_path = os.path.join(source_folder, image_name)
         
        if train == '1':
            shutil.copy(image_path, os.path.join(train_dest, image_name))
            training_size += 1
        elif train == '0':
            shutil.copy(image_path, os.path.join(test_dest, image_name))
            testing_size += 1
            
    print(f"Train Test Split finishes, the training set is {training_size} and the testing set is {testing_size}")

def organize_images(folder_path):

    # Ensure the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Get a list of all files in the specified directory
    files = os.listdir(folder_path)

    # Create folders based on the first three digits of each file name
    for file in files:
        if os.path.isfile(os.path.join(folder_path, file)):
            prefix = file[:3]
            destination_folder = os.path.join(folder_path, prefix)

            # Create folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Move the file to the destination folder
            shutil.move(os.path.join(folder_path, file), os.path.join(destination_folder, file))

    print("Organizing images completed.")

def augment_train_images(): 
    datasets_root_dir = os.path.join("data", "CUB_200_2011", "cub200_cropped")

    dir = os.path.join(datasets_root_dir, "train_cropped")
    target_dir = os.path.join(datasets_root_dir, "train_cropped_augmented")

    makedir(target_dir)
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]
    
    for i in range(len(sorted(folders))):
        fd = folders[i]
        tfd = target_folders[i]

        if os.path.exists(tfd): 
            continue  
        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=os.path.join('..', "..", "train_cropped_augmented", tfd.split("/")[-1]))
        
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=os.path.join('..', "..", "train_cropped_augmented", tfd.split("/")[-1]))
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=os.path.join('..', "..", "train_cropped_augmented", tfd.split("/")[-1]))
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        
        # random_distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory=os.path.join('..', "..", "train_cropped_augmented", tfd.split("/")[-1]))
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in range(10):
           p.process()
        del p


        if i % 100 == 0: 
            print(f"{i}/{len(folders)} augmented.")

    print("Image augmentation, Done")



