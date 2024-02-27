import os
import shutil

from PIL import Image

images_file = 'CUB_200_2011/images.txt'

image_dict = {}  # image to id dict

with open(images_file, "r") as file:
    for line in file:
        # Split the line into id and image name
        id, image_name = line.strip().split()

        # Replace '/' in the image name with '_'
        image_name = image_name.replace('/', '_')

        # Store in the dictionary
        image_dict[image_name] = id

# Print the resulting dictionary
print(image_dict)


split_file = 'CUB_200_2011/train_test_split.txt'

train_test_dict = {}

with open(split_file, "r") as file:
    for line in file:
        # Split the line into id and image name
        id, training = line.strip().split()

        # Store in the dictionary
        train_test_dict[id] = training

# Print the resulting dictionary
print(train_test_dict)



train_dest = 'datasets/cub200_cropped/train_cropped'
test_dest = 'datasets/cub200_cropped/test_cropped'

source_folder = 'CUB_200_2011/cropped_image'

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
    
    img = Image.open(image_path)
    
    if train == '1':
        shutil.copy(image_path, os.path.join(train_dest, image_name))
        training_size += 1
    elif train == '0':
        shutil.copy(image_path, os.path.join(test_dest, image_name))
        testing_size += 1
        
print(f"Train Test Split finishes, the training set is {training_size} and the testing set is {testing_size}")





