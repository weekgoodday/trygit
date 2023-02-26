from __future__ import print_function
import os
import random
import argparse
import torch
import math
import numpy as np
import wandb
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

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    wandb.init(project="CrossPoint", name=args.exp_name)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # xlin: original code
    # train_loader = DataLoader(ShapeNetRender(transform, n_imgs=2), num_workers=0,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset = ShapeNetRender_Update(transform, n_imgs=2)
    train_loader = DataLoader(dataset, num_workers=16, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device(f"cuda:{args.cudaNo}" if args.cuda else "cpu")

    # xlin: init models
    # zht: load models
    outdim = 256
    PATHPOINT="/home/zht/github_play/crosspoint/CrossPoint/checkpoints/crosspoint_dgcnn_cls/models/best_model_epoch52.pth"
    PATHIMG="/home/zht/github_play/crosspoint/CrossPoint/checkpoints/crosspoint_dgcnn_cls/models/img_model_best_epoch52.pth"
    if args.model == 'dgcnn':
        point_model = DGCNN_update(args).to(device)
        point_head = PntHead(args, outdim).to(device)
        net=torch.load(PATHPOINT)
        point_model.load_state_dict(net,strict=False)
    # xlin: only use the dgcnn model, so disable the following model
    # elif args.model == 'dgcnn_seg':  # DGCNN_seg模型分pretrain和正式工作两部分，pretrain只到
    #     point_model = DGCNN_partseg(args).to(device)
    else:
        raise Exception("Not implemented")

    img_model = ResNet_update(resnet50(), feat_dim=2048)
    img_model = img_model.to(device)
    net=torch.load(PATHIMG)
    img_model.load_state_dict(net,strict=False)
    img_head = ImgHead(feat_dim=2048, outdim=outdim).to(device)

    # intact_model = IntactNet(len(dataset))
    H = HNet(len(dataset),64)
    img_dgdt = dgdtNet(128, 64, outdim)
    pnt_dgdt = dgdtNet(128, 64, outdim)
    # zht: put all parameters to device 
    H.to(device)
    img_dgdt.to(device)
    pnt_dgdt.to(device)

    # xlin: finish initializing models


    wandb.watch(point_model)

    if args.resume:  # 输入是False model不会报错
        # xlin: change the model to point_model to avoid warning at xiao's local computer
        point_model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")

    # xlin: set optimizer
    parameters = list(point_model.parameters()) + list(img_model.parameters())

    # if args.use_sgd:  # 输入是False model不会报错
    #     # xlin: change the model to point_model to avoid warning at xiao's local computer
    #     print("Use SGD")
    #     opt = optim.SGD(point_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    # else:  # 默认是这个 parameters是有定义的
    print("Use Adam")
    opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    parameters_head = list(point_head.parameters()) + list(img_head.parameters())
    head_opt = optim.SGD(parameters_head, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    parameters_dgdt = list(pnt_dgdt.parameters()) + list(img_dgdt.parameters())
    dgdt_opt = optim.SGD(parameters_dgdt, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    H_opt = optim.SGD(H.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    # xlin: assume the total epoch is 100, we may need to adjust to the hyper-parameters used below
    # xlin: when epoch is 40 60 80 90, lr will multiply 0.5
    head_scheduler = torch.optim.lr_scheduler.MultiStepLR(head_opt, milestones=[40, 60, 80, 90], gamma=0.5)
    dgdt_scheduler = torch.optim.lr_scheduler.MultiStepLR(dgdt_opt, milestones=[40, 60, 80, 90], gamma=0.5)
    H_scheduler = torch.optim.lr_scheduler.MultiStepLR(H_opt, milestones=[40, 60, 80, 90], gamma=0.5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature=0.1).to(device)
    # xlin: finish setting optim
    lam=1.0
    warmup = 0 #After trial, we don't need warmup part.
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        point_model.train()
        img_model.train()
        point_head.train()
        img_head.train()
        pnt_dgdt.train()
        img_dgdt.train()
        H.train()

        train_losses = AverageMeter()
        train_imid_losses = AverageMeter()
        train_cmid_losses = AverageMeter()
        train_imid1_losses = AverageMeter()
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')

        if epoch < warmup: #warmup don't need
            head_scheduler.step()
            dgdt_scheduler.step()
            H_scheduler.step()
            for i, ((data_t1, data_t2), (imgs0, imgs1), idxs) in enumerate(train_loader):
                data_t1, data_t2, imgs0, imgs1 = data_t1.to(device), data_t2.to(device), imgs0.to(device), imgs1.to(
                    device)
                batch_size = data_t1.size()[0]
                
                head_opt.zero_grad()
                dgdt_opt.zero_grad()
                H_opt.zero_grad()
                data = torch.cat((data_t1, data_t2))
                data = data.transpose(2, 1).contiguous()
                _, point_feats = point_model(data)  # get features from inputs, the function f_theta in ori article
                pnt_prj_feat = point_head(point_feats)  # project the point features, the function g_phi in ori article

                img_feats0 = img_model(imgs0)
                img_feats1 = img_model(imgs1)
                img_prj_feat0 = img_head(img_feats0)
                img_prj_feat1 = img_head(img_feats1)

                point_t1_feats = pnt_prj_feat[:batch_size, :]
                point_t2_feats = pnt_prj_feat[batch_size:, :]
                aver_pnt_feat = (point_t1_feats + point_t2_feats) / 2
                aver_img_feat = (img_prj_feat0 + img_prj_feat1) / 2
                itct_feat = H(idxs)  # get intact features from H
                dgdt_pnt_feat = pnt_dgdt(itct_feat)  # get degradation feature of points
                dgdt_img_feat = img_dgdt(itct_feat)  # get degradation feature of images
                # calculate similarity
                intact_sim = nn.MSELoss()
                loss_crsmodal1 = intact_sim(aver_pnt_feat, dgdt_pnt_feat)
                loss_crsmodal2 = intact_sim(aver_img_feat, dgdt_img_feat)
                loss_cmid = lam*(loss_crsmodal1 + loss_crsmodal2) #multiply lambda
                loss_imid = criterion(point_t1_feats, point_t2_feats)
                loss_imid1 = criterion(img_prj_feat0, img_prj_feat1)
                total_loss = loss_imid + loss_cmid + loss_imid1
                if(i%2==0): #预训练 偶数batch更新g
                    head_opt.zero_grad()
                    loss1 = total_loss
                    loss1.backward() #avoid free parameters
                    head_opt.step()
                else: #奇数batch更新dgdt、H
                    dgdt_opt.zero_grad()
                    H_opt.zero_grad()
                    loss2 = loss_cmid
                    loss2.backward()
                    dgdt_opt.step()
                    H_opt.step()
                
                train_losses.update(total_loss.item(), batch_size)
                train_imid_losses.update(loss_imid.item(), batch_size)
                train_cmid_losses.update(loss_cmid.item(), batch_size)
                train_imid1_losses.update(loss_imid1.item(), batch_size)

                if i % args.print_freq == 0:  # 这是batch数
                    print(
                        'Pretrain: Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, imid1 loss: %.6f, cmid loss: %.6f ' % (
                            epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg,
                            train_imid1_losses.avg, train_cmid_losses.avg))


        else:
            lr_scheduler.step()
            head_scheduler.step()
            dgdt_scheduler.step()
            H_scheduler.step()
            # torch.autograd.set_detect_anomaly(True)
            # tou1 = 2  # first stage: update the project model(g_phi) for the first tou1 epoch
            # tou2 = 2  # second stage: update degradation matrix for the second tou2 epoch
            # tou3 = 1  # third stage: update the intact matrix H for the last tou3 epoch
            for i, ((data_t1, data_t2), (imgs0, imgs1), idxs) in enumerate(train_loader):
                data_t1, data_t2, imgs0, imgs1 = data_t1.to(device), data_t2.to(device), imgs0.to(device), imgs1.to(
                    device)
                batch_size = data_t1.size()[0]
                data = torch.cat((data_t1, data_t2))
                data = data.transpose(2, 1).contiguous()
                _, point_feats = point_model(data)  # get features from inputs, the function f_theta in ori article
                pnt_prj_feat = point_head(point_feats)  # project the point features, the function g_phi in ori article
                img_feats0 = img_model(imgs0)
                img_feats1 = img_model(imgs1)
                img_prj_feat0 = img_head(img_feats0)
                img_prj_feat1 = img_head(img_feats1)
                point_t1_feats = pnt_prj_feat[:batch_size, :]
                point_t2_feats = pnt_prj_feat[batch_size:, :]
                aver_pnt_feat = (point_t1_feats + point_t2_feats) / 2
                aver_img_feat = (img_prj_feat0 + img_prj_feat1) / 2
                itct_feat = H(idxs)  # get intact features from H
                dgdt_pnt_feat = pnt_dgdt(itct_feat)  # get degradation feature of points
                dgdt_img_feat = img_dgdt(itct_feat)  # get degradation feature of images

                # calculate similarity
                intact_sim = nn.MSELoss()
                loss_crsmodal1 = intact_sim(aver_pnt_feat, dgdt_pnt_feat)
                loss_crsmodal2 = intact_sim(aver_img_feat, dgdt_img_feat)
                loss_cmid = lam*(loss_crsmodal1 + loss_crsmodal2) #multiply lambda
                loss_imid = criterion(point_t1_feats, point_t2_feats)
                loss_imid1 = criterion(img_prj_feat0, img_prj_feat1)
                total_loss = loss_imid + loss_cmid + loss_imid1
                if(i%2==0): #偶数batch更新f、g
                    opt.zero_grad()
                    head_opt.zero_grad()
                    loss1 = total_loss
                    loss1.backward() #avoid free parameters
                    opt.step()
                    head_opt.step()
                else: #奇数batch更新dgdt、H
                    dgdt_opt.zero_grad()
                    H_opt.zero_grad()
                    loss2 = loss_cmid
                    loss2.backward()
                    dgdt_opt.step()
                    H_opt.step()
                

                train_losses.update(total_loss.item(), batch_size)
                train_imid_losses.update(loss_imid.item(), batch_size)
                train_cmid_losses.update(loss_cmid.item(), batch_size)
                train_imid1_losses.update(loss_imid1.item(), batch_size)

                if i % args.print_freq == 0:  # 这是batch数
                    print(
                        'Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, imid1 loss: %.6f, cmid loss: %.6f ' % (
                            epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg,
                            train_imid1_losses.avg, train_cmid_losses.avg))

        wandb_log['Train Loss'] = train_losses.avg
        wandb_log['Train IMIDp Loss'] = train_imid_losses.avg
        wandb_log['Train IMIDi Loss'] = train_imid1_losses.avg
        wandb_log['Train CMID Loss'] = train_cmid_losses.avg

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        #在测试前每代保存
        scheduler_dict={
            "opt_scheduler":lr_scheduler,
            "head_scheduler":head_scheduler,
            "dgdt_scheduler":dgdt_scheduler,
            "H_scheduler":H_scheduler
        }
        opt_dict={"opt":opt,
            "head_opt":head_opt,
            "dgdt_opt":dgdt_opt,
            "H_opt":H_opt}
        model_dict={"fp":point_model,
                "fi":img_model,
                "gp":point_head,
                "gi":img_head,
                "dgd_p":pnt_dgdt,
                "dgd_i":img_dgdt,
                "H":H}
        torch.save(opt_dict,f"checkpoints/{args.exp_name}/models/opt.pt") #in case of blackout, we save parameter every epoch
        torch.save(scheduler_dict,f"checkpoints/{args.exp_name}/models/scheduler.pt")
        torch.save(model_dict,f"checkpoints/{args.exp_name}/models/model.pt")
        # Testing 这里的DataLoader是ModelNet40SVM分类数据集
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=True)
        feats_train = []
        labels_train = []
        point_model.eval()
        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[1]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels
        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)
        feats_test = []
        labels_test = []
        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[1]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels
        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        model_tl = SVC(C=0.1, kernel='linear')  # 只是对提的特征进行线性分类
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)  # 可以得到经过训练train之后的model_tl的分数，看提的特征好不好
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model_epoch{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_model_best_epoch{epoch}.pth'.format(epoch=epoch))
            torch.save(img_model.state_dict(), save_img_model_file)
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)
        wandb.log(wandb_log)
    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch{epoch}_last.pth'.format(epoch=epoch))
    torch.save(point_model.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                       'img_model_epoch{epoch}_last.pth'.format(epoch=epoch))
    torch.save(img_model.state_dict(), save_img_model_file)


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
    parser.add_argument('--cudaNo',type=int,default=0,help="using which cuda") #add an argument
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(args.cudaNo) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    train(args, io)