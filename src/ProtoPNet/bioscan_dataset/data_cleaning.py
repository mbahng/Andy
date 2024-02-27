import pandas as pd
import numpy as np

file_path = '/Users/andywang/Desktop/bioscan/BIOSCAN_Insect_Dataset_metadata.tsv'

# Read the TSV file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Display the DataFrame

df.drop(columns=['copyright_institution', 'photographer',
        'author', 'copyright_contact', 'copyright_license', 'copyright_holder',
                 'processid', 'uri', 'nucraw', 'phylum', 'class'], inplace=True)


# Statistics

print(df['family'].value_counts())
print(df['subfamily'].value_counts())
print(df['tribe'].value_counts())
print((df['name'] == df['order']).sum())

df.drop(columns=['subfamily', 'tribe', 'genus',
        'species', 'subspecies', 'name'], inplace=True)


# Small Insect Order Dataset
df_train = df[df['small_insect_order'] == 'train']
df_test = df[df['small_insect_order'] == 'test']

print(df_train['family'].value_counts())
print(df_test['family'].value_counts())

available_family_df = df_train['family'].value_counts(
)[df_train['family'].value_counts() > 100]  # threshold = 100 for family, threshold should be larger as the classification goes broader


print(available_family_df)

available_family = available_family_df.index.tolist()
available_family.remove('not_classified')

print(available_family)

# Class Balance
df_train = df_train[df_train['family'].isin(available_family)]
df_test = df_test[df_test['family'].isin(available_family)]

df_train.drop(columns=['order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
              'large_insect_order', 'medium_insect_order', 'small_insect_order'], inplace=True)
df_test.drop(columns=['order', 'large_diptera_family', 'medium_diptera_family', 'small_diptera_family',
             'large_insect_order', 'medium_insect_order', 'small_insect_order'], inplace=True)


print(df_train)


np.random.seed(42)


def random_sample(group, sample_size=100):
    if len(group) < sample_size:
        print(f"{group['family'].iloc[0]} does not have enough samples")

    return group.sample(n=min(sample_size, len(group)), random_state=42)


df_train = df_train.groupby('family', group_keys=False).apply(
    lambda x: random_sample(x, 100))
df_test = df_test.groupby('family', group_keys=False).apply(
    lambda x: random_sample(x, 20))

print(df_train)
print(df_test)

df_train.to_csv('/Users/andywang/Desktop/bioscan/bioscan_train.csv', index=False)
df_test.to_csv('/Users/andywang/Desktop/bioscan/bioscan_test.csv', index=False)