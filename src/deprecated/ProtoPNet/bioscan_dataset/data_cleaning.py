import pandas as pd
import numpy as np
import os 

MINBALANCE_COUNT = 100
# nucraw contains genetic information

file_path = os.path.join("data", "BIOSCAN-1M", "BIOSCAN_Insect_Dataset_metadata.tsv")
df = pd.read_csv(file_path, sep='\t')
df.drop(columns=['copyright_institution', 'photographer', 'author', 'copyright_contact', 'copyright_license', 'copyright_holder', 'processid', 'uri', 'phylum', 'class', 'subfamily', 'tribe', 'genus', 'species', 'subspecies', 'name', 'order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family'], inplace=True)


df.drop(columns=['copyright_institution', 'photographer',
        'author', 'copyright_contact', 'copyright_license', 'copyright_holder',
                 'processid', 'uri', 'nucraw', 'phylum', 'class'], inplace=True)
df.drop(columns=['subfamily', 'tribe', 'genus',
        'species', 'subspecies', 'name'], inplace=True)

# Small Insect Order Dataset
df_train = df[df['small_insect_order'] == 'train']
df_test = df[df['small_insect_order'] == 'test']

# drop all classes that have image counts below MINBALANCE_COUNT 
available_family_df = df_train['family'].value_counts(
)[df_train['family'].value_counts() > MINBALANCE_COUNT]  # threshold = MINBALANCE_COUNT for family, threshold should be larger as the classification goes broader

available_family = available_family_df.index.tolist()
available_family.remove('not_classified')

# Class Balance
df_train = df_train[df_train['family'].isin(available_family)]
df_test = df_test[df_test['family'].isin(available_family)]

df_train.drop(columns=['order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
              'large_insect_order', 'medium_insect_order', 'small_insect_order'], inplace=True)
df_test.drop(columns=['order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
             'large_insect_order', 'medium_insect_order', 'small_insect_order'], inplace=True)




np.random.seed(42)


def random_sample(group, sample_size=MINBALANCE_COUNT):
    if len(group) < sample_size:
        print(f"{group['family'].iloc[0]} does not have enough samples")

    return group.sample(n=min(sample_size, len(group)), random_state=42)


df_train = df_train.groupby('family', group_keys=False).apply(
    lambda x: random_sample(x, MINBALANCE_COUNT))
df_test = df_test.groupby('family', group_keys=False).apply(
    lambda x: random_sample(x, 20))

print(df_train)
print(df_test)

df_train.to_csv('/Users/andywang/Desktop/bioscan/bioscan_train.csv', index=False)
df_test.to_csv('/Users/andywang/Desktop/bioscan/bioscan_test.csv', index=False)
