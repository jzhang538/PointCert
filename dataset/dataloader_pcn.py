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
import torch
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0, keepdim=True)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def pc_batch_normalize(pc):
    centroid = torch.mean(pc,1).unsqueeze(1)
    pc = pc - centroid
    dis = torch.sqrt(torch.sum(pc**2, 2))
    m = torch.max(dis, 1)[0]
    m = m.view(-1,1,1)
    pc = pc / m
    return pc


class ModelNetDataLoader(Dataset):
    def __init__(self, args, split):
        self.root = args.data_dir
        self.num_group = args.num_group
        self.use_ensemble = args.use_ensemble
        self.split = split

        self.num_input = args.num_input         # e.g., 32
        self.num_recon = args.num_recon       # e.g., 1024

        path = os.path.join(self.root,'modelnet40_{}_{}_{}.dat'.format(self.split,'md5',self.num_group))
        if not os.path.exists(path):
            print('File does not exists at %s. Error!!!' % path)
            exit()
        else:
            print('Load %s point clouds from %s.' % (self.split, path))
            with open(path, 'rb') as f:
                self.list_of_points, self.list_of_labels, self.list_of_groups = pickle.load(f)
        self.l = len(self.list_of_points)
        self.current_group_pointers = np.zeros((self.l)) # pointers of current group in each point cloud 
    
    def __len__(self):
        return self.l

    def __getitem__(self, index):
        points, label, groups = self.list_of_points[index], self.list_of_labels[index], self.list_of_groups[index]

        if self.split=='customer':
            group_idx = self.current_group_pointers[index]
        if self.split=='test' :
            if self.use_ensemble:
                group_idx = self.current_group_pointers[index]
            else:
                group_idx = 0 # use group 0 of each testing point cloud to monitor the testing accuracy of base point cloud classifier
        selected_indices = self.list_of_groups[index] == group_idx
        selected_points = points[selected_indices]
        pcn_input = resample_pcd(selected_points, self.num_input) # PCN assumes a fixed number of points as inputs

        pcn_input = pcn_input[:, 0:3]
        coarse_gt = points[:self.num_recon, 0:3]
        return torch.from_numpy(pcn_input), torch.from_numpy(coarse_gt), torch.from_numpy(label), index