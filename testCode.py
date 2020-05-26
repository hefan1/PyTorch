import os
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preTrainChannelGroupNet import CGNN
from CUB_loader import CUB200_loader
from AIRCRAFT_loader import AIR100_loader
from CARS_loader import CARS196_loader
from CompactBilinearPooling import CountSketch, CompactBilinearPooling
from GCN_STN_BLN import GTBNN

def main():
    net=torch.nn.DataParallel(GTBNN()).cuda()
    net.load_state_dict(torch.load('GCN_STN_BLN_DiffLr.pth'))
    net.eval()
    testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split='test')
    test_loader = data.DataLoader(testset, batch_size=1,
                                  shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    num_correct = 0
    num_total = 0
    for X, y in test_loader:
        # Data.
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda())
        X.requires_grad = True
        # Prediction.
        score, cgs = net(X)
        _, prediction = torch.max(score.data, 1)
        num_total += y.size(0)
        num_correct += torch.sum(prediction == y.data).item()
    print('Test accuracy is %.2f%%'%(100 * num_correct / num_total))

if __name__ =='__main__':
    main()