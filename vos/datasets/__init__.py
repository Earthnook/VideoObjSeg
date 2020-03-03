""" a collection of wrappers to deal with datasets
Each usable dataset output a single data contained in a dictionary.
For this problem formulation, the dictionary contains at least 3 items:
    video: a tensor with shape (T, C, H, W)
    mask: a tensor with shape (T, N, H, W) N for multi-objects
    n_objects: a int telling this video has number of objects to track

NOTE: N has to be no smaller than n_objects+1. And it has to be invariant
to index in order to make data batch
"""