
from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
#import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from torch.utils.data import DataLoader

from datasets.data import ShapeNetRender, ModelNet40SVM, ShapeNetRender_Update
from models.dgcnn import *
from util import IOStream, AverageMeter
from models.intactnn import HNet, dgdtNet, IntactNet
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    #wandb.init(project="CrossPoint", name=args.exp_name)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # xlin: original code
    # train_loader = DataLoader(ShapeNetRender(transform, n_imgs=2), num_workers=0,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset = ShapeNetRender_Update(transform, n_imgs=2)
    train_loader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:1" if args.cuda else "cpu")

    # xlin: init models
    # Try to load models
    outdim = 256
    if args.model == 'dgcnn':
        point_model = DGCNN_update(args).to(device)
        point_head = PntHead(args, outdim).to(device)
    # xlin: only use the dgcnn model, so disable the following model
    # elif args.model == 'dgcnn_seg':  # DGCNN_seg模型分pretrain和正式工作两部分，pretrain只到
    #     point_model = DGCNN_partseg(args).to(device)
    else:
        raise Exception("Not implemented")
    PATH_save="/home/zht/github_play/crosspoint/CrossPoint/scheduler.pt"
    net=torch.load(PATH_save)
    for key in net:
        print(key)
        print(net[key])

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',  # 默认exp会和文件夹名有关
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')  # 输入是100
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')  # store_true代表有这个参数就True 不过和resume一样 都没有
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')  # 输入是15
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')  # 改成了5
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')  # 200指的是每200个batch打印一次
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:  # evaluate是False 只会运行train
        train(args, io)
    else:  # test在别的地方运行
        test(args, io)
