"""
Author: Zhang
Date: March 2023
"""
import os
import warnings
import pickle
import hashlib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataLoader(Dataset):
    def __init__(self, args, split):
        self.root = args.data_dir
        self.npoints = args.num_point
        self.num_group = args.num_group
        self.split = split

        path = os.path.join(self.root,'modelnet40_{}_{}_{}.dat'.format(self.split,'md5',self.num_group))
        if not os.path.exists(path):
            print('File does not exists at %s. Error!!!' % path)
            exit()
        else:
            print('Load %s point clouds from %s.' % (self.split, path))
            with open(path, 'rb') as f:
                self.list_of_points, self.list_of_labels, self.list_of_groups = pickle.load(f)
    
    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index):
        points, label, groups = self.list_of_points[index], self.list_of_labels[index], self.list_of_groups[index]
        sampled_points = points[:self.npoints]
        sampled_points[:, 0:3] = pc_normalize(sampled_points[:, 0:3])
        return sampled_points[:, 0:3], label[0], index

    def collate_fn(self, samples):
        bs =len(samples)

        # get max group_npoints
        max_group_npoints = 0
        for (points, label, _) in samples:
            if len(points)>max_group_npoints:
                max_group_npoints = len(points)

        points_batch = np.zeros((bs, max_group_npoints, 3))
        label_batch = np.zeros((bs))
        index_batch = np.zeros((bs))
        for sampleId, (points, label, index) in enumerate(samples):
            points_batch[sampleId, :len(points), :3] = points
            label_batch[sampleId] = label
            index_batch[sampleId] = index
        return points_batch, label_batch, index_batch