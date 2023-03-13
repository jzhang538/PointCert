"""
Author: Zhang
Date: March 2023
"""
import os
import sys
BASE_DIR = os.getcwd()
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import random
import torch
import numpy as np
import time
import datetime
import logging
import utils
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from dataset.dataloader_hashed import ModelNetDataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
setup_seed(2023)


def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')

    parser.add_argument('--num_group', type=int, default=400, help='number of groups')
    parser.add_argument('--classifier_path', type=str, default='./scenarioII/provider/best_model_m_400.pth')
    parser.add_argument('--data_dir', type=str, default='./processed', help='path to preprocessed data')
    parser.add_argument('--use_ensemble', action='store_true', default=False, help='use ensemble during testing')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader):
    classifier = model.eval()
    num_correct = 0
    num_total = 0
    for j, (points, target, _) in tqdm(enumerate(loader), total=len(loader)):
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        target = torch.Tensor(target)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        num_correct += correct.item()
        num_total += points.size()[0]
    acc = float(num_correct/num_total)
    return acc


def ensemble_test(model, loader, dataset, save_path, num_group=400, num_class=40):
    num_test_case = len(dataset)
    votes = np.zeros((num_test_case, num_class))
    print(votes.shape)

    classifier = model.eval()
    dataset.current_group_pointers = np.zeros((num_test_case))
    num_correct = 0
    num_total = 0

    ### Repeat 'num_group' times to predict a label for each group in each testing point cloud ###
    for i in tqdm(range(num_group)):
    # for i in tqdm(range(50)):
        for j, (points, target, indices) in enumerate(loader):
            ### Update Pointers ###
            indices = indices.astype(np.int32)
            dataset.current_group_pointers[indices]+=1
            dataset.current_group_pointers[indices] = dataset.current_group_pointers[indices]%num_group

            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            target = torch.Tensor(target).long()
            indices = indices.astype(np.int32)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += points.size()[0]
            for k in range(len(pred_choice)):
                votes[indices[k],pred_choice[k]]+=1
    print("Average Accuracy:", float(num_correct/num_total))

    sorted_votes = np.sort(votes, axis=1) # ascending order

    ### Calculate certified accuracy under no attack ###
    # res_arr: [target, label_maximum_vote, label_second_maximum_vote, maximum_vote, second_maximum_vote]
    res_arr = np.zeros((num_test_case,5))
    num_correct = 0
    num_total = 0
    for j, (_, target, indices) in enumerate(loader):
        target = target.astype(np.int32)
        indices = indices.astype(np.int32)
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
        num_correct += correct.item()
        num_total += len(target)
    acc_no_attack = float(num_correct/num_total)
    print('Certified/Empirical accuracy under no attack: %f' % (acc_no_attack))

    ### Save results to calculate certified accuracy curve ###
    np.save(save_path, res_arr)


def main(args):
    ### Setup Environments ###
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    ### Load Test Dataset ###
    print('Load dataset')
    test_dataset = ModelNetDataLoader(args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)


    ### Load Model ###
    print('Load model')
    num_class = args.num_category
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    try:
        checkpoint = torch.load(args.classifier_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrained checkpoint.')
    except:
        start_epoch = 0
        print('No existing checkpoint, starting training from scratch.')


    ### Test or Ensemble Test ###
    if not args.use_ensemble:
        with torch.no_grad():
            acc = test(classifier.eval(), testDataLoader)
            print("Testing accuracy on single group:", acc)
    else:
        if not os.path.exists('./curves/'):
            os.makedirs('./curves/')
        save_path='./curves/scenarioII_{}.npy'.format(args.num_group)
        ensemble_test(classifier, testDataLoader, test_dataset, save_path)
        print("Results save at {}.".format(save_path))

if __name__ == '__main__':
    args = parse_args()
    main(args)