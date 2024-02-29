"""
The datasets return an array of taxonomy values for each label.
taxonomy_level_array array controls the order of those classes.
taxonomy_level_index_map will give you the index for each.
"""
taxonomy_level_array = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]
taxonomy_level_index_map = {taxonomy_level_array[i]: i for i in range(len(taxonomy_level_array))}
