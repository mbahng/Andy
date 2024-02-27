import os
import shutil
from PIL import Image


# Source directory
source_directory = 'CUB_200_2011/images/'

# Destination directory
destination_directory = 'CUB_200_2011/images_copy/'


os.makedirs(destination_directory, exist_ok=True)

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


images_file = 'CUB_200_2011/images.txt'


image_dict = {}

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


bounding_box_file = 'CUB_200_2011/bounding_boxes.txt'

bounding_boxes_dict = {}

with open(bounding_box_file, "r") as file:
    for line in file:
        # Split the line into id and bounding box coordinates
        id, *bbox_coordinates = line.strip().split()

        # Convert bounding box coordinates to tuple of floats
        bounding_box = tuple(map(float, bbox_coordinates))

        # Store in the dictionary
        bounding_boxes_dict[id] = bounding_box

# Print the resulting dictionary
print(bounding_boxes_dict)


image_folder = "CUB_200_2011/images_copy"
output_folder = "CUB_200_2011/cropped_image"


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

print(f"{counter} images cropped")





