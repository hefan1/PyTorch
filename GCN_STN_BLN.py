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

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class GTBNN(torch.nn.Module):
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
        ######################### Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features  # fine tune?
        self.features = torch.nn.Sequential(*list(self.features.children())
        [:-22])  # Remove pool2 and rest, lack of computational resource
        # No grad for convVGG
        # for param in self.features.parameters():
        #     param.requires_grad = False

        #################### Channel Grouping Net
        self.fc1_ = torch.nn.Linear(128*28*28, 64)#lack of resource
        self.fc2_ = torch.nn.Linear(128*28*28, 64)
        self.fc3_ = torch.nn.Linear(128*28*28, 64)

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(64, 128)

        # global grad for hook
        self.image_reconstruction = None
        self.register_hooks()

        ################### STN input N*3*448*448
        self.localization = [
                nn.Sequential(
                nn.MaxPool2d(4,stride=4),#112
                nn.ReLU(True),

                nn.Conv2d(3, 32, kernel_size=5,stride=1,padding=2),  # 112
                nn.MaxPool2d(2, stride=2),  # 56
                nn.ReLU(True),

                nn.Conv2d(32, 48, kernel_size=3,stride=1,padding=1),
                nn.MaxPool2d(2, stride=2),  # 56/2=28
                nn.ReLU(True),

                nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2),  # 28/2=14
                nn.ReLU(True) #output 64*14*14
            ).cuda(),
            nn.Sequential(
                nn.MaxPool2d(4, stride=4),  # 112
                nn.ReLU(True),

                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 112
                nn.MaxPool2d(2, stride=2),  # 56
                nn.ReLU(True),

                nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2),  # 56/2=28
                nn.ReLU(True),

                nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2),  # 28/2=14
                nn.ReLU(True)  # output 64*14*14
            ).cuda(),
            nn.Sequential(
                nn.MaxPool2d(4, stride=4),  # 112
                nn.ReLU(True),

                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 112
                nn.MaxPool2d(2, stride=2),  # 56
                nn.ReLU(True),

                nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2),  # 56/2=28
                nn.ReLU(True),

                nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2),  # 28/2=14
                nn.ReLU(True)  # output 64*14*14
            ).cuda()
        ]
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = [
                nn.Sequential(
                nn.Linear(64 * 14 * 14, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda(),
            nn.Sequential(
                nn.Linear(64 * 14 * 14, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda(),
            nn.Sequential(
                nn.Linear(64 * 14 * 14, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda()
        ]
        # Initialize the weights/bias with identity transformation
        for fc_locx in self.fc_loc:
            fc_locx[2].weight.data.zero_()
            fc_locx[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        ########################Bilinear CNN output 256 channels
        self.bcnnConv=torch.nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())
                                            [:-15])  # Remove pool3 and rest.
        #BCNN Linear classifier.
        self.fc = torch.nn.Linear(256 ** 2, 200)
        torch.nn.init.kaiming_normal(self.fc.weight.data)  # 何凯明初始化
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)  # fc层的bias进行constant初始化

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            # 在全局变量中保存输入图片的梯度，该梯度由第一层卷积层
            # 反向传播得到，因此该函数需绑定第一个 Conv2d Layer
            self.image_reconstruction = grad_in[0]

        # 获取 module，
        modules = list(self.features.named_children())

        # # 遍历所有 module，对 ReLU 注册 forward hook 和 backward hook
        # for name, module in modules:
        #     if isinstance(module, nn.ReLU):
        #         module.register_forward_hook(forward_hook_fn)
        #         module.register_backward_hook(backward_hook_fn)

        # 对第1层卷积层注册 hook
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def weightByGrad(self, Xi,Xo):
        XiSum=torch.sum(Xi,dim=1)
        XiSum.backward(torch.ones(XiSum.shape).cuda(),retain_graph=True)
        #normalize, not tried......
        gradImg= self.image_reconstruction.data#[0].permute(1, 2, 0)
        gradImg=torch.sqrt(gradImg*gradImg)

        self.zero_grad()

        res=gradImg*Xo
        return res


    # Spatial transformer network forward function
    def stn(self, x,i):
        xs = self.localization[i](x)
        xs = xs.view(-1, 64 * 14 * 14)
        theta = self.fc_loc[i](xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta,torch.Size([x.size()[0], x.size()[1], 96, 96]))# x.size())
        x = F.grid_sample(x, grid)

        return x


    def BCNN_N(self,X):#parameter calculate undone!
        # N = X.size()[0]
        # assert X.size() == (N, 3, 448, 448)
        # X=nn.MaxPool2d(kernel_size=8, stride=8).cuda()(X)
        # X=nn.Linear(3*56*56,200).cuda()(X.view(-1,3*56*56))
        #
        # assert X.size() == (N, 200)
        # return X

        N = X.size()[0]
        assert X.size() == (N, 3, 96, 96)
        X = self.bcnnConv(X)
        assert X.size() == (N, 256, 24, 24)
        X = X.view(N, 256, 24 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (24 ** 2)  # Bilinear
        assert X.size() == (N, 256, 256)
        X = X.view(N, 256 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X


    def forward(self, Xo):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = Xo.size()[0]
        assert Xo.size() == (N, 3, 448, 448)
        X = self.features(Xo)
        assert X.size() == (N, 128, 224, 224)
        Xp = nn.MaxPool2d(kernel_size=8, stride=8)(X)
        assert Xp.size() == (N, 128, 28, 28)
        Xp = Xp.view(-1, 128 * 28 * 28)
        # 3 way, get attention mask
        X1 = F.relu(self.fc1_(Xp))
        X2 = F.relu(self.fc2_(Xp))
        X3 = F.relu(self.fc3_(Xp))
        X1 = self.fc1(X1)
        X2 = self.fc2(X2)
        X3 = self.fc3(X3)
        # multiple mask elementwisely, get 3 attention part
        X1 = X1.unsqueeze(dim=2).unsqueeze(dim=3) * X
        X2 = X2.unsqueeze(dim=2).unsqueeze(dim=3) * X
        X3 = X3.unsqueeze(dim=2).unsqueeze(dim=3) * X
        #get the graduate w.r.t input image and multiple, then X1 become N*3*448*448
        X1=self.weightByGrad(X1,Xo)
        X2=self.weightByGrad(X2,Xo)
        X3=self.weightByGrad(X3,Xo)
        # use stn to crop, size become (N,3,96,96)
        X1 = self.stn(X1, 0)
        X2 = self.stn(X2, 1)
        X3 = self.stn(X3, 2)
        #3 BCNN 3 size==(N,200)
        X1=self.BCNN_N(X1)
        X2=self.BCNN_N(X2)
        X3=self.BCNN_N(X3)
        #sum them up, for the predict max
        res=X1+X2+X3

        return res

def main():
    # tmpNet=torch.nn.DataParallel(CGNN()).cuda()
    # tmpNet.load_state_dict(torch.load("preTrainedGCNetModel.pth"))
    # tmpNet.eval()
    # state_dict=tmpNet.state_dict()
    # print(state_dict)
    #####test code
    net = torch.nn.DataParallel(GTBNN()).cuda()
    print(net)
    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
    train_loader = data.DataLoader(trainset, batch_size=2,
                                   shuffle=True, collate_fn=trainset.CUB_collate, num_workers=1)  # shuffle?
    criterion = torch.nn.CrossEntropyLoss()
    solver = torch.optim.SGD(
        net.parameters(), lr=1,
        momentum=0.9, weight_decay=1e-5)
    for X, y in train_loader:
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda())
        solver.zero_grad()
        # Forward pass.
        X.requires_grad=True
        score = net(X)
        loss = criterion(score, y)
        # epoch_loss.append(loss.data[0])
        # Prediction.
        _, prediction = torch.max(score.data, 1)
        print(torch.sum(prediction == y.data),flush=True)
        loss.backward()
        solver.step()
        print(loss)


if __name__ == '__main__':
    main()