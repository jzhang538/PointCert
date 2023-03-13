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
import argparse
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def getHexStrMod(hex_str, mod_num):
    str_len=len(hex_str)
    result_mod=0
    for idx,ch in enumerate(hex_str):
        result_mod = (result_mod*16 + int(ch, 16)) % mod_num
    return result_mod

def hash_point(point):
    point_str = str(point[0])+str(point[1])+str(point[2])
    hex_str = hashlib.md5(point_str.encode()).hexdigest()
    return hex_str

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids

def hash_point_clouds(root, split, num_group, save_dir):
    '''
    Data Format
    list_of_points: [[10000×3], [10000×3], ...]
    list_of_labels: [label1, label2, ...]
    list_of_groups: [[10000], [10000], ...]
    '''
    # load files
    catfile = os.path.join(root, 'modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    shape_ids = {}
    shape_ids = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_{}.txt'.format(split)))]
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
    datapaths = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[i]) + '.txt') for i
                     in range(len(shape_ids))]
    print('The size of %s point clouds is %d.' % (split, len(datapaths)))


    save_path = os.path.join(save_dir,'modelnet40_{}_{}_{}.dat'.format(split,'md5',num_group))
    if not os.path.exists(save_path):
        print("Start preprocissing %s point clouds." % (split))
        list_of_points = [None] * len(datapaths)
        list_of_labels = [None] * len(datapaths)
        list_of_groups = [None] * len(datapaths)
        desired_groups = 0
        total_groups = 0

        # hash each point cloud
        for index in tqdm(range(len(datapaths)), total=len(datapaths)):
            fn = datapaths[index]
            cls = classes[datapaths[index][0]]
            cls = np.array([cls]).astype(np.int32)
            points = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            assert len(points)==10000
            
            counters = np.zeros((num_group))
            groups = np.zeros((len(points)))
            for k,point in enumerate(points):
                hex_str = hash_point(point)
                result_mod = getHexStrMod(hex_str, num_group)
                groups[k] = result_mod
                counters[result_mod]+=1

                # # naive
                # point_sum = point[0]+point[1]+point[2]
                # s = float("%.3f" % (point_sum))
                # s = int(s*1e3)
                # result_mod = s%self.num_group
                # assert result_mod>=0 and result_mod<self.num_group
                # groups[k] = result_mod
                # counters[result_mod]+=1
            assert np.sum(counters)==len(points)

            list_of_points[index] = points
            list_of_labels[index] = cls
            list_of_groups[index] = groups 

            # statistics
            mean = len(points)/num_group
            for cnt in counters:
                if cnt<(mean+10) and cnt>(mean-10):
                    desired_groups+=1
            total_groups += num_group
        print("Proportion of desired groups:", desired_groups/total_groups)

        with open(save_path, 'wb') as f:
            pickle.dump([list_of_points, list_of_labels, list_of_groups], f)
        print("Finished.")
    else:
        print('File exists at %s.' % save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_group', type=int, default=400)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    root = './data/modelnet40_normal_resampled'
    save_dir = './processed'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_group = args.num_group
    print("Divide each point cloud into {} sub-point clouds.".format(num_group))

    # hash provider point clouds
    hash_point_clouds(root, 'provider', num_group, save_dir)
    print('')
    # hash customer point clouds
    hash_point_clouds(root, 'customer', num_group, save_dir)
    print('')
    # hash test point clouds
    hash_point_clouds(root, 'test', num_group, save_dir)
    print('')