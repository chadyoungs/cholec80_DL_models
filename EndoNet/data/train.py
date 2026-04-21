# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:09:49 2020
@author: xiaoxiaoyang
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from vgg_model import VGG
from dataloader import Dataset
import config

    
# train and test
def train(supertrial_index):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(config.epoches):
        # loss
        train_loss = 0.0

        # correctness
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            #initialization
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
        
            _, outputs = net(inputs)

            # loss
            loss = criterion(outputs, labels)

            # correctness
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum()

            # optimization
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            if batch_idx % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, train_loss/1000))
                train_loss = 0.0

        print('Saving epoch %d model ...' % (epoch + 1))
        print("supertrial_index: %s Current epoch, Accuracy of the network on the train images: %d %%" % (supertrial_index, 100 * correct / total))
        
        state = { 'net': net.state_dict(),
                  'epoch':epoch + 1}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/supertrial_%s' % supertrial_index):
            os.mkdir('checkpoint/supertrial_%s' % supertrial_index)
        torch.save(state, './checkpoint/supertrial_%s/jigsaws_epoch_%d.ckpt' % (supertrial_index, epoch + 1))

    print('Finished Training')

########################
# test
########################
def test(supertrial_index):
    print('Using the last epoch')
    checkpoint = torch.load('./checkpoint/supertrial_%s/jigsaws_epoch_%d.ckpt' % (supertrial_index, config.epoches))
    net.load_state_dict(checkpoint['net'])
    net.to(device)

    start_epoch = checkpoint['epoch']      

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            _, outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum()

    cur_correct = 100 * correct / total
    print("Accuracy of the network on the test images: %d %%" % cur_correct)

    # confusion matrix initialization
    y_cm = []
    y_p_cm = []
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            _, outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted.cpu().numpy() == labels.cpu().numpy()).squeeze()

            # confusion matrix
            labels_cm = labels.cpu().numpy().tolist()
            for i in labels_cm:
                y_cm.append(i)
            predicted_cm = predicted.cpu().numpy().tolist()
            for i in predicted_cm:
                y_p_cm.append(i)
                
            for i in range(len(labels.numpy())):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("confusion matrix")
    print(confusion_matrix(np.array(y_cm), np.array(y_p_cm)))
    
    for i in range(10):
        if class_total[i] == 0:
            class_total[i] += 1
            print("%5s : None exists" % (gesture_classes[i])) 
        print("supertrial_index: %s Accuracy of %5s : %2d %%" %(supertrial_index, gesture_classes[i], 100 * class_correct[i] / class_total[i]))

    return cur_correct


if __name__ == "__main__":
    # gesture classes
    gesture_classes = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
    classes_count = config.net_params['out_dim']

    test_scores = []
    for supertrial_index in range(config.super_trail_count):
        dataset_params = config.dataset_params
        net = VGG(**config.net_params)

        # cuda option
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        net.to(device)

        # multi-GPU training
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print('using{} GPUs to train'.format(device_count))
            net = nn.DataParallel(net)

    
        dataloaders = {}
        for name in ['Train', 'Test']:
            dataloaders[name] = DataLoader(Dataset(name, supertrial_index), 
                                   batch_size = dataset_params['batch_size'], 
                                   shuffle = dataset_params['shuffle'])
        trainloader = dataloaders['Train']
        testloader = dataloaders['Test']
        if supertrial_index > 1:        
            train(supertrial_index)
            test_score = test(supertrial_index)
            test_scores.append(test_score)

    print('final test accuracy equals to %4d %%' % (sum(test_scores)/len(test_scores)))
