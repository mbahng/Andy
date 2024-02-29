import pandas as pd
import numpy as np
import os
from shutil import copyfile


train_file_path = '/Users/andywang/Desktop/bioscan/bioscan_train.csv'
test_file_path = '/Users/andywang/Desktop/bioscan/bioscan_test.csv'

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

base_directory = '/Users/andywang/Desktop/bioscan/images/cropped_256/'


def organize_images(df, base_directory, train=True):
    # Create base directory if it doesn't exist
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Group DataFrame by 'family' column
    grouped_df = df.groupby('family')

    for family, group in grouped_df:

        # Create a folder for each family under the base directory

        if train:
            family_directory = os.path.join(
                base_directory, 'train', str(family))
        else:
            family_directory = os.path.join(
                base_directory, 'test', str(family))

        if not os.path.exists(family_directory):
            os.makedirs(family_directory)

        # Iterate through rows in the family group
        for index, row in group.iterrows():
            # Get image path from the 'image' column
            image_path = os.path.join(base_directory,
                'part' + str(row['chunk_number']), row['image_file'])

            # Copy the image to the family folder
            destination_path = os.path.join(
                family_directory, row['image_file'])
            copyfile(image_path, destination_path)


organize_images(df_train, base_directory)
organize_images(df_test, base_directory, train=False)

print("Organization Done")
