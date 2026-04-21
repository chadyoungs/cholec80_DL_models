# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:03:56 2021
@author: ThinkPad

Pytorch Test
"""

import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F


# logsoft-max + NLLLoss
m = nn.LogSoftmax()
loss = nn.NLLLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)

print('logsoftmax + nllloss output is {}\n'.format(output))

# crossentropyloss
loss = nn.CrossEntropyLoss()
# input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
#target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(input, target)
print('crossentropy output is {}\n'.format(output))


# one hot label loss
C = 5
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
print('target is {}'.format(target))

N = target.size(0)
# N 是batch-size大小
# C is the number of classes.
labels = torch.full(size=(N, C), fill_value=0)
print('labels shape is {}'.format(labels.shape))
labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)
print('labels is {}'.format(labels))

log_prob = torch.nn.functional.log_softmax(input, dim=1)
loss = -torch.sum(log_prob * labels) / N
print('N is {}'.format(N))
print('one-hot loss is {}'.format(loss))


'''
m = nn.Sigmoid()
loss = nn.BCELoss()
input = autograd.Variable(torch.randn(3), requires_grad=True)
target = autograd.Variable(torch.FloatTensor(3).random_(2))
output = loss(m(input), target)
output.backward()
'''