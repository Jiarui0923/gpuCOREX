import numpy as np

def convert_batch(x, partition, pad=True):
    partition = np.array(partition).T
    full_size = np.max(partition[1] - partition[0])
    result = []
    for p in partition.T:
        fragment = x[p[0] - 1:p[1]]
        if pad:
            if   len(fragment.shape) == 1: fragment = np.concatenate([fragment, np.zeros([full_size-(p[1]-p[0])])])
            elif len(fragment.shape) == 2: fragment = np.concatenate([fragment, np.zeros([full_size-(p[1]-p[0]), fragment.shape[1]])])
            else: raise ValueError
        result.append(fragment)
    if pad:
        result = np.array(result)
    return result