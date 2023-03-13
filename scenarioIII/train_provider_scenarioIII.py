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
from dataset.dataloader_standard import ModelNetDataLoader


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
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--num_group', type=int, default=400, help='number of groups')

    parser.add_argument('--num_point', type=int, default=1024, help='number of points')
    parser.add_argument('--exp_dir', type=str, default='./scenarioIII/provider', help='experiment root')
    parser.add_argument('--data_dir', type=str, default='./processed', help='path to preprocessed data')
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


def main(args):
    ### Setup Environments and Directories ###
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True)


    ### Load Provider and Test Datasets ###
    print('Load dataset')
    train_dataset = ModelNetDataLoader(args=args, split='provider')
    test_dataset = ModelNetDataLoader(args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
             num_workers=args.num_workers, drop_last=True, collate_fn=train_dataset.collate_fn)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)


    ### Load Model and Optimizer ###
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
        checkpoint = torch.load(str(exp_dir) + '/best_model_standard_{}.pth'.format(args.num_point))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrained checkpoint.')
    except:
        start_epoch = 0
        print('No existing checkpoint, starting training from scratch.')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)
    

    ### Initial Testing Accuracy ###
    global_epoch = 0
    global_step = 0
    best_acc = 0.0
    with torch.no_grad():
        acc = test(classifier.eval(), testDataLoader)
        best_acc = acc
        print("Initial performance:", acc)

    
    ### Training Standard Point Cloud Classifier on the Provider Side ###
    print('Start training')
    for epoch in range(start_epoch, args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        scheduler.step()

        num_correct = 0
        num_total = 0
        for batch_id, (points, target,_) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # print(points.shape)
            optimizer.zero_grad()

            points = utils.random_point_dropout(points)
            points[:, :, 0:3] = utils.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = utils.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            target = torch.Tensor(target)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            num_correct += correct.item()
            num_total += points.size()[0]
            loss.backward()
            optimizer.step()
            global_step += 1

        ### Training Accuracy ###
        train_acc = float(num_correct/num_total)
        print('Train Instance Accuracy: %f' % train_acc)

        ### Testing Accuracy at Each Epoch ###
        with torch.no_grad():
            acc = test(classifier.eval(), testDataLoader)
            if (acc >= best_acc):
                best_acc = acc
                best_epoch = epoch + 1
            print('Test Instance Accuracy: %f; Best Instance Accuracy: %f' % (acc, best_acc))

            ### For simplicity, we save the best model for evaluation ###
            ### Feel free to change to the final model ###
            if (acc >= best_acc):
                savepath = str(exp_dir) + '/best_model_standard_{}.pth'.format(args.num_point)
                print('Save model at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'acc': acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        
        global_epoch += 1
    
    savepath = str(exp_dir) + '/final_model_{}.pth'.format(args.num_point)
    print('Save final model at %s' % savepath)
    state = {
        'epoch': best_epoch,
        'acc': acc,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
    print('End of training')

if __name__ == '__main__':
    args = parse_args()
    main(args)