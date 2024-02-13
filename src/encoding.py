import numpy as np

dna_one_hot_mapping = dict(zip("ACGT", range(4))) 

def one_hot_encode(dna_seq):
    one_hot = [dna_one_hot_mapping[a] for a in dna_seq]
    return np.eye(4)[one_hot].transpose()

def k_mer_freq(dna_seq, k=4):
    one_hot = one_hot_encode(dna_seq) # 4 by n
    
    freq = np.sum(one_hot[:, :k], axis=1)
    encoding = [freq.copy()]
    
    for i in range(k, len(one_hot[0])):
        freq += one_hot[:, i] - one_hot[:, i-k]
        encoding.append(freq.copy())
    
    return np.transpose(encoding)

cgt_mapping = {"A": np.array([0,0]),
               "C": np.array([0,1]),
               "G": np.array([1,1]),
               "T": np.array([1,0])}

def cgr_encoding(dna_seq):
    prev_point = np.array([0.5,0.5])
    
    points = [prev_point]
    
    for a in dna_seq:
        point = 0.5 * (prev_point + cgt_mapping[a])
        points.append(point)
        prev_point = point
    
    return np.array(points)