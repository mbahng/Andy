import os
import shutil

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

# Example usage: Replace 'your_folder_path' with the actual path of your image folder
organize_images('datasets/cub200_cropped/train_cropped')
organize_images('datasets/cub200_cropped/test_cropped')