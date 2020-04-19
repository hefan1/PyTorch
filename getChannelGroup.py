import os

import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from CUB_loader import CUB200_loader


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class preCHANNEL(torch.nn.Module):
    """B-CNN for CUB200.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features#fine tune?
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5. Output (N, 512, 28, 28)


    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)

        # print("fuck?",flush=False)

        X=X.transpose(0,1)
        ret = [[] for i in range(512)]
        for i,c in zip(X,range(512)):
            ret[c]=[(j.argmax()//28,j.argmax()%28) for j in i] # calculate the positions of highest response

        return ret


net=preCHANNEL()
batch_size=4

def getPosition():
    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=False, collate_fn=trainset.CUB_collate, num_workers=1)  # shuffle?
    ret=[[] for i in range(512)]
    with torch.no_grad():
        cnt=0
        for X, y in train_loader:
            pos=net(X)
            cnt+=X.size()[0]
            # if(cnt%10==0):
            # print(cnt,flush=False)
            for i,c in zip(pos,range(512)):
                for j in i:
                    ret[c].append(j[0].item())
                    ret[c].append(j[1].item())

    return ret

def main():
    ret=getPosition()
    res=pd.DataFrame(data=ret)
    res.to_csv('channelPeakPos.csv');

if __name__ == '__main__':
    main()
