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
    parser.add_argument('--loss_func', type=str, default='cd')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_ensemble', action='store_true', default=False, help='use ensemble')

    parser.add_argument('--num_category', type=int, default=40)
    parser.add_argument('--classifier_name', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--classifier_path', default='./scenarioIII/provider/best_model_standard_1024.pth')
    parser.add_argument('--exp_dir', type=str, default='./scenarioIII/customer/')
    parser.add_argument('--data_dir', type=str, default='./processed')

    parser.add_argument('--num_group', type=int, default=400)
    parser.add_argument('--num_input', type=int, default=32)
    parser.add_argument('--num_recon', type=int, default=1024)
    parser.add_argument('--l', type=float, default=0.0005) # lambda
    parser.add_argument('--ratio', type=float, default=0.25) # label ratio
    return parser.parse_args()


def main(args):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cd_loss = ChamferDistance()
    emd_loss = EarthMoverDistance()
    pcn_loss = cd_loss if args.loss_func == 'cd' else emd_loss


    ### Load Customer and Test Datasets ###
    print('Load dataset')
    train_dataset = ModelNetDataLoader(args=args, split='customer')
    test_dataset = ModelNetDataLoader(args=args, split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
             num_workers=args.num_workers, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.num_workers)


    ### Load PCN Model and Optimizer ###
    network = AutoEncoder()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    pcn_path = os.path.join(args.exp_dir, 'best_model_m_{}_input_{}_recon_{}.pth'.format(args.num_group, args.num_input, args.num_recon))
    if os.path.exists(pcn_path):
        print('Use pretrained checkpoint at {}.'.format(pcn_path))
        network.load_state_dict(torch.load(pcn_path))
    else:
        print('No existing checkpoint, starting training from scratch.')
    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)


    ### Load Provider Point Cloud Classifier ###
    num_class = args.num_category
    classifier = importlib.import_module(args.classifier_name).get_model(num_class).cuda()
    checkpoint = torch.load(args.classifier_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()


    ### Generate/Load Label Flags ###
    label_flags_pth = './processed/customer_label_flags_{}.npy'.format(args.ratio)
    if not os.path.exists(label_flags_pth):
        idx_ls = np.random.permutation(len(train_dataset))
        selected_l = int(args.ratio*len(train_dataset))
        selected_idx = idx_ls[:selected_l]
        all_label_flags = np.zeros(len(train_dataset)) 
        all_label_flags[selected_idx] = 1 # Indicate whether each customer point cloud is labeled or not
        np.save(label_flags_pth, all_label_flags) 
    else:
        all_label_flags = np.load(label_flags_pth)
    print(args.ratio, np.sum(all_label_flags), len(all_label_flags))


    ### Start Training ###
    print("Length of dataset:", len(train_dataset))
    max_iter = int(len(train_dataset) / args.batch_size + 0.5)
    maximum_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        ### Randomly Initialize the 'current_group_pointers' of Training Point Clouds ###
        if epoch%args.num_group==0:
            train_dataset.current_group_pointers = np.random.randint(args.num_group, size=len(train_dataset))
        network.train()
        total_loss, total_pc_loss, iter_count = 0, 0, 0
        
        num_correct = 0
        num_total = 0
        for i, data in enumerate(train_dataloader, 1):
            ### Make Pointers Point to Next Sub-point-clouds ###
            pcn_inputs, coarse_gt, target, indices  = data
            train_dataset.current_group_pointers[indices]+=1
            train_dataset.current_group_pointers[indices] = train_dataset.current_group_pointers[indices]%args.num_group
            optimizer.zero_grad()

            label_flags = all_label_flags[indices]
            pcn_inputs = pcn_inputs.to(DEVICE)
            coarse_gt = coarse_gt.to(DEVICE)
            target = target.to(DEVICE).squeeze()
            pcn_inputs = pcn_inputs.permute(0, 2, 1)

            ### Point Completion ###
            v, coarse_pred = network(pcn_inputs)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            loss_pc = pcn_loss(coarse_gt, coarse_pred)

            ### Point Classification ###
            coarse_pred = pc_batch_normalize(coarse_pred)
            coarse_pred = coarse_pred.permute(0, 2, 1)
            pred, _ = classifier(coarse_pred)
            labeled_indices = (label_flags==1)
            loss_cls = F.nll_loss(pred[labeled_indices], target[labeled_indices].long())
            # loss_cls = F.nll_loss(pred, target.long())

            ### Classification Accuracy ###
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += len(target)

            ### Optimization ###
            loss = loss_pc + loss_cls*args.l
            clip_grad_norm_([p for group in optimizer.param_groups \
                                     for p in group['params']], 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pc_loss += loss_pc.item()
            iter_count += 1

        acc = num_correct/num_total
        scheduler.step()
        print("Training epoch {}/{}: avg completion loss = {}, avg loss = {}, accuracy = {}.".format(epoch, args.epochs,\
            total_pc_loss / iter_count, total_loss / iter_count, acc))


        # Evaluation
        network.eval()
        with torch.no_grad():
            total_loss, iter_count = 0, 0
            num_correct = 0
            num_total = 0
            for i, data in enumerate(test_dataloader, 1):
                pcn_inputs, coarse_gt, target, indices = data
                pcn_inputs = pcn_inputs.to(DEVICE)
                coarse_gt = coarse_gt.to(DEVICE)
                target = target.to(DEVICE).squeeze()
                pcn_inputs = pcn_inputs.permute(0, 2, 1)
                
                ### Point Completion ###
                v, coarse_pred = network(pcn_inputs)
                coarse_pred = coarse_pred.permute(0, 2, 1)
                loss = pcn_loss(coarse_gt, coarse_pred)
                total_loss += loss.item()
                iter_count += 1

                ### Point Classification ###
                coarse_pred = pc_batch_normalize(coarse_pred)
                coarse_pred = coarse_pred.permute(0, 2, 1)
                pred, _ = classifier(coarse_pred)

                ### Classification Accuracy ###
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                num_correct += correct.item()
                num_total += len(target)

            mean_loss = total_loss / iter_count
            mean_acc = num_correct/num_total
            print("Validation epoch {}/{}, avg completion loss = {}, accuracy = {}.".format(epoch, args.epochs, mean_loss, mean_acc))

            ### For simplicity, we save the best model for evaluation ###
            ### Feel free to change to the final model ###
            if mean_acc > maximum_acc:
                best_epoch = epoch
                maximum_acc = mean_acc
                torch.save(network.state_dict(), pcn_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
