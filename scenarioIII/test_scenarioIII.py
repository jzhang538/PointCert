"""
Author: Zhang
Date: March 2023
"""
import argparse
import time
import importlib
import sys
import os
BASE_DIR = os.getcwd()
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import random
from matplotlib import pyplot as plt
from dataset.dataloader_pcn import ModelNetDataLoader, pc_batch_normalize
from models.autoencoder import AutoEncoder
from distance.chamfer_distance import ChamferDistanceFunction
from distance.emd_module import emdFunction
from utils import *


### PCN Losses ###
class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()
    
    def forward(self, pcs1, pcs2):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        dist1, dist2 =  ChamferDistanceFunction.apply(pcs1, pcs2)  # (B, N), (B, M)
        dist1 = torch.mean(torch.sqrt(dist1))
        dist2 = torch.mean(torch.sqrt(dist2))
        return (dist1 + dist2) / 2

class EarthMoverDistance(nn.Module):
    def __init__(self, eps=0.005, max_iter=3000):
        super(EarthMoverDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
    
    def forward(self, pcs1, pcs2):
        dist, _ = emdFunction.apply(pcs1, pcs2, self.eps, self.max_iter)
        return torch.sqrt(dist).mean()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
setup_seed(2020)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--loss_func', type=str, default='cd')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_ensemble', action='store_true', default=False, help='use ensemble')

    parser.add_argument('--num_category', type=int, default=40)
    parser.add_argument('--classifier_name', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--classifier_path', default='./scenarioIII/provider/best_model_standard_1024.pth')
    parser.add_argument('--pcn_path', type=str, default='./scenarioIII/customer/best_model_m_400_input_32_recon_1024.pth')
    parser.add_argument('--data_dir', type=str, default='./processed')

    parser.add_argument('--num_group', type=int, default=400)
    parser.add_argument('--num_input', type=int, default=32)
    parser.add_argument('--num_recon', type=int, default=1024)
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cd_loss = ChamferDistance()
    emd_loss = EarthMoverDistance()
    pcn_loss = cd_loss if args.loss_func == 'cd' else emd_loss


    ### Load Customer Dataset ###
    print('Load dataset')
    test_dataset = ModelNetDataLoader(args=args, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.num_workers)


    ### Load PCN Model ###
    network = AutoEncoder()
    ckpt_path = os.path.join(args.pcn_path)
    if os.path.exists(ckpt_path):
        print('Use pretrained checkpoint at {}.'.format(ckpt_path))
        network.load_state_dict(torch.load(ckpt_path))
    else:
        print('No existing checkpoint, starting training from scratch.')
    network.to(DEVICE)
    network.eval()


    ### Load Provider Point Cloud Classifier ###
    num_class = args.num_category
    classifier = importlib.import_module(args.classifier_name).get_model(num_class).cuda()
    checkpoint = torch.load(args.classifier_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()


    # Test or Ensemble Test
    if not args.use_ensemble:
        test(test_dataloader, network, classifier, pcn_loss)
        test_gt(test_dataloader, network, classifier, pcn_loss)
    else:
        if not os.path.exists('./curves/'):
            os.makedirs('./curves/')
        save_path='./curves/scenarioIII_{}.npy'.format(args.num_group)
        ensemble_test(network, classifier, test_dataloader, test_dataset, save_path, args.num_group, num_class)


# Evaluation using Reconstructed Coarse Point Cloud
def test(dataLoader, network, classifier, loss_fuc):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        total_loss, iter_count = 0, 0
        num_correct = 0
        num_total = 0
        for i, data in tqdm(enumerate(dataLoader, 1), total=len(dataLoader)):
            pcn_inputs, coarse_gt, target, indices  = data
            pcn_inputs = pcn_inputs.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            target = target.to(DEVICE).squeeze()

            pcn_inputs = pcn_inputs.permute(0, 2, 1)
            v, coarse_pred = network(pcn_inputs)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            loss = loss_fuc(coarse_gt, coarse_pred)
            total_loss += loss.item()

            coarse_pred = pc_batch_normalize(coarse_pred)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            # coarse_gt = pc_batch_normalize(coarse_gt)
            # coarse_gt = coarse_gt.permute(0, 2, 1)

            pred, _ = classifier(coarse_pred)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += len(target)
            iter_count += 1

        mean_acc = num_correct/num_total
        mean_loss = total_loss / iter_count
        print("Reconstruction loss is {}.".format(mean_loss))
        print("Evaluation accuracy is {}".format(mean_acc))


# Evaluation using Groudtruth Coarse Point Cloud
def test_gt(dataLoader, network, classifier, loss_fuc):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        total_loss, iter_count = 0, 0
        num_correct = 0
        num_total = 0
        for i, data in tqdm(enumerate(dataLoader, 1), total=len(dataLoader)):
            pcn_inputs, coarse_gt, target, indices  = data
            pcn_inputs = pcn_inputs.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            target = target.to(DEVICE).squeeze()

            pcn_inputs = pcn_inputs.permute(0, 2, 1)
            v, coarse_pred = network(pcn_inputs)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            loss = loss_fuc(coarse_gt, coarse_pred)
            total_loss += loss.item()

            # coarse_pred = pc_batch_normalize(coarse_pred)
            # coarse_pred = coarse_pred.permute(0, 2, 1)
            coarse_gt = pc_batch_normalize(coarse_gt)
            coarse_gt = coarse_gt.permute(0, 2, 1)
            
            pred, _ = classifier(coarse_gt)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += len(target)
            iter_count += 1

        mean_acc = num_correct/num_total
        mean_loss = total_loss / iter_count
        print("Reconstruction loss is {}.".format(mean_loss))
        print("Evaluation accuracy is {}".format(mean_acc))


def ensemble_test(network, classifier, loader, dataset, save_path, num_group=400, num_class=40):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_test_case = len(dataset)
    votes = np.zeros((num_test_case, num_class))
    print(votes.shape)

    network = network.eval()
    classifier = classifier.eval()
    dataset.current_group_pointers = np.zeros((num_test_case))
    num_correct = 0
    num_total = 0

    ### Repeat 'num_group' times to predict a label for each group in each testing point cloud ###
    for i in tqdm(range(num_group)):
    # for i in tqdm(range(10)):
        for i, data in enumerate(loader):
            pcn_inputs, coarse_gt, target, indices = data
            dataset.current_group_pointers[indices]+=1
            dataset.current_group_pointers[indices] = dataset.current_group_pointers[indices]%num_group

            pcn_inputs = pcn_inputs.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            target = target.to(DEVICE).squeeze()
            indices = np.array(indices).reshape(-1).astype(np.int32)

            pcn_inputs = pcn_inputs.permute(0, 2, 1)
            v, coarse_pred = network(pcn_inputs)
            coarse_pred = coarse_pred.permute(0, 2, 1)

            coarse_pred = pc_batch_normalize(coarse_pred)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            pred, _ = classifier(coarse_pred)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += len(target)
            for k in range(len(pred_choice)):
                votes[indices[k],pred_choice[k]]+=1
    print("Average Accuracy:", float(num_correct/num_total))

    sorted_votes = np.sort(votes, axis=1) # ascending order

    ### Calculate certified accuracy under no attack ###
    # res_arr: [target, label_maximum_vote, label_second_maximum_vote, maximum_vote, second_maximum_vote]
    res_arr = np.zeros((num_test_case,5))
    num_correct = 0
    num_total = 0
    for j, (_, _, target, indices) in enumerate(loader):
        target = np.array(target).reshape(-1).astype(np.int32)
        indices = np.array(indices).reshape(-1).astype(np.int32)
        batch_votes = votes[indices]

        for k, index in enumerate(indices):
            ### No tie ###
            if sorted_votes[index,-1]!=sorted_votes[index,-2]:
                max_indice = np.argwhere(votes[index]==sorted_votes[index,-1])[0]
                second_max_indice = np.argwhere(votes[index]==sorted_votes[index,-2])[0]
            ### Tie occurs
            else: 
                max_indice = np.argwhere(votes[index]==sorted_votes[index,-1])[0]
                second_max_indice = np.argwhere(votes[index]==sorted_votes[index,-2])[1]
            res_arr[index][0] = target[k]
            res_arr[index][1] = max_indice
            res_arr[index][2] = second_max_indice
            res_arr[index][3] = sorted_votes[index, -1]
            res_arr[index][4] = sorted_votes[index, -2]

        pred_choice = np.argmax(batch_votes,-1)
        correct = np.sum(pred_choice == target)
        num_correct += correct
        num_total += len(target)
    acc_no_attack = float(num_correct/num_total)
    print('Certified/Empirical accuracy under no attack: %f' % (acc_no_attack))

    ### Save results to calculate certified accuracy curve ###
    np.save(save_path, res_arr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
