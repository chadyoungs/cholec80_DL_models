import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict

import argparse

from model import AlexNet, EasyFCNet
from dataloader import Dataset
import config


def train_on_epochs(train_loader, test_loader, restore_from):
    # configuraion settings of training process
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # instantiating computation graph model
    model1 = AlexNet(init_weights=True, freezing=False, **config.net_params)
    model1.to(device)
    model2 = EasyFCNet(init_weights=True, freezing=False, **config.net_params)
    model2.to(device)

    # multi-GPU training
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('using{} GPUs to train'.format(device_count))
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)

    ckpt = {}
    # training from check point
    if restore_from is not None:
        ckpt = torch.load(restore_from)
        model1.load_state_dict(ckpt['model1_state_dict'])
        print('Model1 is loaded from %s' % (restore_from))
        model2.load_state_dict(ckpt['model2_state_dict'])
        print('Model2 is loaded from %s' % (restore_from))
    
    # extracting the params of net, pre-training
    model_params1 = [{'params':model1.parameters(), 'lr':config.learning_rate_classifier}
                     ]
    model_params2 = [{'params':model2.module.fc_phase.parameters(), 'lr':config.learning_rate_classifier}
                     ]
    
    # optimizer
    optimizer1 = torch.optim.SGD(model_params1, lr=config.learning_rate_feature, momentum=config.momentum)
    optimizer2 = torch.optim.SGD(model_params2, lr=config.learning_rate_feature)

    if restore_from is not None:
        optimizer1.load_state_dict(ckpt['optimizer1_state_dict'])
        optimizer2.load_state_dict(ckpt['optimizer1_state_dict'])

    # training informations
    info = defaultdict(list)

    start_ep = ckpt['epoch'] + 1 if 'epoch' in ckpt else 0

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Start to train
    for ep in range(start_ep, config.epoches):
        train_losses, train_scores1, train_scores2 = train(model1, model2, train_loader, optimizer1, optimizer2, ep, device)
        test_loss, test_score1, test_score2 = validation(model1, model2,test_loader, optimizer1, optimizer2, ep, device)

        # Saving the  infos
        info['train_losses'].append(train_losses)
        info['train_scores1'].append(train_scores1)
        info['train_scores2'].append(train_scores2)
        info['test_loss'].append(test_loss)
        info['test_score1'].append(test_score1)
        info['test_score2'].append(test_score2)

        # Saving the model
        ckpt_path = os.path.join(save_path, 'ep-%d.pth' % ep)
        if (ep + 1) % config.save_interval == 0:
            torch.save({
                'epoch': ep,
                'model1_state_dict': model1.state_dict(),
                'model2_state_dict': model1.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer1.state_dict()
            }, ckpt_path)
            print('Model of Epoch %3d has been saved to: %s' % (ep, ckpt_path))

    with open('train_info.json', 'w') as f:
        json.dump(info, f)

    print('the end of training process')


def train(model1, model2, dataloader, optimizer1, optimizer2, epoch, device):
    model1.train()
    model2.train()

    train_losses = []
    train_scores1, train_scores2 = [], []

    print('Size of Training Set: ', len(dataloader.dataset))

    for i, (X, y1, y2) in enumerate(dataloader):
        # intializing the optimizer
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        # forward
        X1, y1_ = model1(X.to(device)) 
        X1 = torch.cat([X1, y1_], dim=1)
        y2_ = model2(X1) 
        
        # calculate the loss
        criterion1 = nn.MultiLabelSoftMarginLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
        
        loss1 = criterion1(y1_, y1.to(device))
        loss2 = criterion2(y2_, y2.to(device).long())
        
        loss = loss1 + loss2
  
        y1_ = y1_.cpu().detach().numpy()
        y1_ = np.array(y1_ > config.threshold, dtype=float)
        y2_ = y2_.argmax(dim=1)
        
        acc1 = accuracy_score(y1_, y1.cpu().detach().numpy())
        acc2 = accuracy_score(y2_.cpu().detach().numpy(), y2.cpu().detach().numpy())

        # backward 
        loss.sum().backward()
        optimizer1.step() 
        optimizer2.step() 
        
        # saving the loss infos, and the acc is tool detect acc actually
        train_losses.append(loss.sum().item()/X.shape[0])
        train_scores1.append(acc1)
        train_scores2.append(acc2)

        if (i + 1) % config.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d: acc1 = %.2f, acc2 = %.2f, loss = %.2f' % (epoch, i + 1, len(dataloader), acc1, acc2, loss.sum().item()/X.shape[0]))

    return np.mean(train_losses), np.mean(train_scores1), np.mean(train_scores2)


def validation(model1, model2, test_loader, optimizer1, optimizer2, epoch, device):
    model1.eval()
    model2.eval()

    print('Size of Test Set: ', len(test_loader.dataset))

    # prepared to certify the model on train set
    test_loss = 0
    test_scores1, test_scores2 = [], []
    
    #groud truth and predictions
    y1_gd, y2_gd = [], []
    y1_pred, y2_pred = [], []

    # no backward
    with torch.no_grad():
        for X, y1, y2 in tqdm(test_loader, desc='Validating'):
            # prediction on test set
            y1, y2 = y1.to(device), y2.to(device)
            
            y1_, y2_ = model1(X)
            X1, y1_ = model1(X.to(device)) 
            X1 = torch.cat([X1, y1_], dim=1)
            y2_ = model2(X1)

            # calculate the loss
            criterion1 = nn.BCELoss()
            criterion2 = nn.CrossEntropyLoss()
        
            loss1 = criterion1(y1_, y1)
            loss2 = criterion2(y2_, y2.long())
            
            loss = loss1 + loss2
            
            test_loss += loss.sum().item()/X.shape[0]
            
            y1_ = y1_.cpu().detach().numpy()
            y1_ = np.array(y1_ > config.threshold, dtype=float)
            y2_ = y2_.argmax(dim=1)
        
            y1_gd += y1.cpu().numpy().tolist()
            y1_pred += y1_.tolist()
            y2_gd += y2.cpu().numpy().tolist()
            y2_pred += y2_.cpu().numpy().tolist()
            
            acc1 = accuracy_score(y1_, y1.cpu().detach().numpy())
            acc2 = accuracy_score(y2_.cpu().detach().numpy(), y2.cpu().detach().numpy())
            
            # saving the loss infos, and the acc is tool detect acc actually
            test_scores1.append(acc1)
            test_scores2.append(acc2)

    # calculate the loss
    test_loss /= len(test_loader)
     
    # only tool detection
    print('[Epoch %3d]Test avg loss: %0.4f, acc1: %0.2f, acc2: %0.2f\n' % (epoch, test_loss, np.mean(test_scores1), np.mean(test_scores2)) )

    return test_loss, np.mean(test_scores1), np.mean(test_scores2)


def parse_args():
    parser = argparse.ArgumentParser(usage='python train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--dataset_path', help='path to your datasets', default='/media/ExtHDD/cholec80_data')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path

    # prepared dataloader
    dataloaders = {}
    for name in ['train', 'test']:
        dataloaders[name] = DataLoader(Dataset(name), 
                                       batch_size = config.dataset_params['batch_size'], 
                                       shuffle = config.dataset_params['shuffle'])
    train_on_epochs(dataloaders['train'], dataloaders['test'], args.restore_from)
