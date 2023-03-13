"""
Author: Zhang
Date: March 2023
"""
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import random
from matplotlib import pyplot as plt
import matplotlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--scenario', type=str, default='II')
    parser.add_argument('--num_group', type=int, default=400, help='number of groups')
    parser.add_argument('--tau', type=int, default=1, help='1 for point addition/deletion; 2 for point modification/perturbation')
    return parser.parse_args()


def params_init():
    font = {'family' : 'serif',
                'size'   : 30,
                }
    matplotlib.rc('font', **font)
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['figure.figsize'] = 12,8
    params = {'legend.fontsize': 30}  
    plt.rcParams.update(params)


def get_certified_perturbation_size(res_arr, tau):
    '''
    Args:
        res_arr: [num_test_case, 5] 
        tau: 1 for point addition/deletion attack; 2 for point modification/perturbation attack
    Returns:
        certified_perturbation_sizes: [num_test_case]
    '''
    # res_arr: [target, label_maximum_vote, label_second_maximum_vote, maximum_vote, second_maximum_vote]

    certified_perturbation_sizes = []
    for i in range(len(res_arr)):
        assert res_arr[i,1]!=res_arr[i,2]
        certified_perturbation_size = int((res_arr[i,3] - (res_arr[i,4] + int(res_arr[i,1]>res_arr[i,2])))/(2*tau)) 
        certified_perturbation_sizes.append(certified_perturbation_size)
    return certified_perturbation_sizes


def get_certified_accuracy_curve(res_arr, num_group, tau=1):
    '''
    Args:
        res_arr: [num_test_case, 5]
        num_group: number of group
    Returns:
        certified_accuracy_curve
    '''
    certified_perturbation_sizes = get_certified_perturbation_size(res_arr, tau)

    xlim = int(num_group/(2*tau))+50
    step = 10
    certified_accuracy_curve = []
    for j in range(0,xlim,step):
        num_certified = 0
        for i in range(len(certified_perturbation_sizes)):
            if certified_perturbation_sizes[i]>=j and res_arr[i, 0]==res_arr[i,1]: 
                num_certified+=1
        certified_accuracy_curve.append([j, float(num_certified/len(certified_perturbation_sizes))])
        if j==num_group/(2*tau): # for visualization 
            certified_accuracy_curve.append([num_group/(2*tau)+1, 0])
    
    return np.array(certified_accuracy_curve)


def draw_certified_accuracy_curve(certified_accuracy_curve, scenario, save_path):
    xlim = certified_accuracy_curve[-1, 0]

    # Visualization
    params_init()
    fig, axes = plt.subplots()
    color_list = ['orange']
    label_list = ['PointCert ({})'.format(scenario)]
    marker_list = ['o']

    axes.set_title('Point Addition Attacks')
    axes.set_xlabel("Number of Added Points, t")
    axes.set_ylabel('Certified Accuracy@t')
    axes.grid()
    axes.set_xlim(0,xlim)
    axes.set_ylim(0,1)
    axes.plot(certified_accuracy_curve[:,0], certified_accuracy_curve[:,1], color=color_list[0],
                     label=label_list[0], linewidth=5, marker=marker_list[0])
    plt.legend()
    # plt.show()

    pp = PdfPages(save_path)
    pp.savefig()
    pp.close()
    plt.close()
    print("File is saved at:", save_path)


def main(args):
    ### Load Results ###
    res_arr_path = './curves/scenario{}_{}.npy'.format(args.scenario, args.num_group)
    res_arr = np.load(res_arr_path)
    
    ### Calculate Certified Accuracy Curve ###
    certified_accuracy_curve = get_certified_accuracy_curve(res_arr, num_group=args.num_group, tau=args.tau)
    print('')
    print("{}\t{}".format('Perturbation Size', 'Certified Accuracy'), flush=True)
    for i in range(0,len(certified_accuracy_curve),1):
        print("{}\t\t\t{}".format(certified_accuracy_curve[i, 0], certified_accuracy_curve[i, 1]), flush=True)
    print('')

    ### Draw Certified Accuracy Curve ###
    draw_path = './curves/scenario{}_{}.pdf'.format(args.scenario, args.num_group)
    draw_certified_accuracy_curve(certified_accuracy_curve, args.scenario, draw_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
