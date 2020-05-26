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


import logging
def get_log(file_name):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)



    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger=get_log('./data/GCN_STN_BLN_DiffLr.log')
# logger=get_log('test.log')

resultGradImg=None

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class GTBNN(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        ######################### Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features  # fine tune?
        self.features = torch.nn.Sequential(*list(self.features.children())
        [:-22])  # Remove pool2 and rest, lack of computational resource

        #################### Channel Grouping Net
        self.fc1 = torch.nn.Linear(128*28*28, 128)
        self.fc2 = torch.nn.Linear(128*28*28, 128)
        self.fc3 = torch.nn.Linear(128*28*28, 128)


        torch.nn.init.kaiming_normal_(self.fc1.weight.data, nonlinearity='relu')# kaiming initiate
        if self.fc1.bias is not None:
            torch.nn.init.constant_(self.fc1.bias.data, val=0)  # fc bias constant initiate
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, nonlinearity='relu')
        if self.fc2.bias is not None:
            torch.nn.init.constant_(self.fc2.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.fc3.weight.data, nonlinearity='relu')
        if self.fc3.bias is not None:
            torch.nn.init.constant_(self.fc3.bias.data, val=0)

        # self.fc1_ = torch.nn.Linear(128, 128*16)#lack of resource
        # self.fc2_ = torch.nn.Linear(128, 128*16)
        # self.fc3_ = torch.nn.Linear(128, 128*16)
        #
        # torch.nn.init.kaiming_normal_(self.fc1_.weight.data, nonlinearity='relu')
        # if self.fc1_.bias is not None:
        #     torch.nn.init.constant_(self.fc1_.bias.data, val=0)  # fc层的bias进行constant初始化
        # torch.nn.init.kaiming_normal_(self.fc2_.weight.data, nonlinearity='relu')
        # if self.fc2_.bias is not None:
        #     torch.nn.init.constant_(self.fc2_.bias.data, val=0)  # fc层的bias进行constant初始化
        # torch.nn.init.kaiming_normal_(self.fc3_.weight.data, nonlinearity='relu')
        # if self.fc3_.bias is not None:
        #     torch.nn.init.constant_(self.fc3_.bias.data, val=0)  # fc层的bias进行constant初始化


        # global grad for hook
        self.layerNorm = nn.LayerNorm([224, 224])
        self.image_reconstruction = None
        self.register_hooks()
        self.GradWeight=1e-1


        # ################### STN input N*3*448*448
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
                nn.ReLU(True) #output 64*14*14 64*7*7
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
                nn.Linear(64 *7 *7, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda(),
            nn.Sequential(
                nn.Linear(64 * 7 * 7, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda(),
            nn.Sequential(
                nn.Linear(64 * 7 * 7, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            ).cuda()
        ]
        # Initialize the weights/bias with identity transformation
        for fc_locx in self.fc_loc:
            fc_locx[2].weight.data.zero_()
            fc_locx[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        ########################Bilinear CNN output 256 channels
        self.bcnnConv_1=torch.nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())
                                            [:-1])  # Remove pool5.
        self.bcnnConv_2 = torch.nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())
                                            [:-1])  # Remove pool5.
        self.bcnnConv_3 = torch.nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())
                                            [:-1])  # Remove pool5.
        #BCNN Linear classifier.
        self.bfc1 = torch.nn.Linear(512*512, 200)
        self.bfc2 = torch.nn.Linear(512*512, 200)
        self.bfc3 = torch.nn.Linear(512*512, 200)
        torch.nn.init.kaiming_normal_(self.bfc1.weight.data)  # kaiming initiate
        if self.bfc1.bias is not None:
            torch.nn.init.constant_(self.bfc1.bias.data, val=0)  # fc bias constant initiate
        torch.nn.init.kaiming_normal_(self.bfc2.weight.data)
        if self.bfc2.bias is not None:
            torch.nn.init.constant_(self.bfc2.bias.data, val=0)
        torch.nn.init.kaiming_normal_(self.bfc3.weight.data)
        if self.bfc3.bias is not None:
            torch.nn.init.constant_(self.bfc3.bias.data, val=0)


    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            # save grad of input images in global variables
            self.image_reconstruction = grad_in[0]

        # get module，
        modules = list(self.features.named_children())

        # iterate all modules register forward hook and backward hook
        # for name, module in modules:
        #     if isinstance(module, nn.ReLU):
        #         module.register_forward_hook(forward_hook_fn)
        #         module.register_backward_hook(backward_hook_fn)

        # register hook for the first conv layer
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def weightByGrad(self, Xi,Xo):
        XiSum=torch.sum(Xi,dim=1)
        XiSum.backward(torch.ones(XiSum.shape).cuda(),retain_graph=True)
        #normalize, not tried......
        gradImg= self.image_reconstruction.data#[0].permute(1, 2, 0)
        gradImg=torch.sqrt(gradImg*gradImg)#needed
        gradImg=self.layerNorm(gradImg)

        self.zero_grad()


        res=gradImg*self.GradWeight*Xo+Xo
        return res


    # Spatial transformer network forward function
    def stn(self, x,i):
        xs = self.localization[i](x)
        xs = xs.view(-1, 64 * 7 * 7)
        theta = self.fc_loc[i](xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta,torch.Size([x.size()[0], x.size()[1], 96, 96]))# x.size())
        x = F.grid_sample(x, grid)

        return x


    def BCNN_N(self,X,bcnnConv,fc):#parameter calculate undone!
        N = X.size()[0]
        # assert X.size() == (N, 3, 224, 224)
        X = nn.Dropout2d(p=0.5)(bcnnConv(X))
        # X = bcnnConv(X)#overfitting
        # assert X.size() == (N, 512, 14, 14)#224/16

        X = X.view(N, 512, 6 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (6 ** 2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = fc(X)

        assert X.size() == (N, 200)
        return X


    def forward(self, Xo):
        N = Xo.size()[0]
        # assert Xo.size() == (N, 3, 448, 448)
        X = self.features(Xo)
        # assert X.size() == (N, 128, 112, 112)
        Xp = nn.MaxPool2d(kernel_size=4, stride=4)(X)
        # Xp = F.adaptive_avg_pool2d(X, (1, 1))
        # assert Xp.size() == (N, 128, 28, 28)
        Xp = Xp.view(-1, 128*28*28 )
        # 3 way, get attention mask
        X1 = self.fc1(Xp)
        X2 = self.fc2(Xp)
        X3 = self.fc3(Xp)
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
        X1=self.BCNN_N(X1,self.bcnnConv_1,self.bfc1)
        X2=self.BCNN_N(X2,self.bcnnConv_2,self.bfc2)
        X3=self.BCNN_N(X3,self.bcnnConv_3,self.bfc3)
        #sum them up, for the predict max
        res=X1+X2+X3

        return res


def main():
    net = torch.nn.DataParallel(GTBNN()).cuda()
    print(net)
    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
    testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split='test')
    train_loader = data.DataLoader(trainset, batch_size=16,
                                   shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)  # shuffle?
    test_loader = data.DataLoader(testset, batch_size=16,
                                  shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    criterion = torch.nn.CrossEntropyLoss()

    conv_params = list(map(id, net.module.features.parameters()))#Don't learn convNet

    # gcn_params = list(map(id, net.module.fc1_.parameters())) \
    #              + list(map(id, net.module.fc2_.parameters())) \
    #              + list(map(id, net.module.fc3_.parameters())) \
    #              + list(map(id, net.module.fc1.parameters())) \
    #              + list(map(id, net.module.fc2.parameters())) \
    #              + list(map(id, net.module.fc3.parameters()))

    gcn_params =   list(map(id, net.module.fc1.parameters())) \
                 + list(map(id, net.module.fc2.parameters())) \
                 + list(map(id, net.module.fc3.parameters()))

    stn_params =   list(map(id, net.module.localization[0].parameters())) \
                 + list(map(id, net.module.localization[1].parameters())) \
                 + list(map(id, net.module.localization[2].parameters())) \
                 + list(map(id, net.module.fc_loc[0].parameters())) \
                 + list(map(id, net.module.fc_loc[1].parameters())) \
                 + list(map(id, net.module.fc_loc[2].parameters()))

    bcn_fc_params = list(map(id, net.module.bfc1.parameters())) \
                    + list(map(id, net.module.bfc2.parameters())) \
                    + list(map(id, net.module.bfc3.parameters()))

    bcn_conv_params = list(
        filter(lambda p: id(p) not in gcn_params + conv_params + bcn_fc_params, net.module.parameters()))


    gcn_params = list(filter(lambda p: id(p)  in gcn_params, net.module.parameters()))

    stn_params = list(filter(lambda p: id(p)  in stn_params, net.module.parameters()))

    bcn_fc_params = list(filter(lambda p: id(p)  in bcn_fc_params, net.module.parameters()))

    solver = torch.optim.SGD([
        {'params': gcn_params, 'lr': 0.01},
        {'params': bcn_fc_params, 'lr': 0.01},
        {'params': stn_params, 'lr': 1.0}
    ], lr=0.001, momentum=0.9, weight_decay=1e-8)
    lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        solver, mode='max', factor=0.2, patience=3, verbose=True,
        threshold=1e-4)
    # solver = torch.optim.Adam(net.parameters(),lr=0.3,weight_decay=1e-4)
    # lrscheduler=torch.optim.lr_scheduler.CosineAnnealingLR(solver,T_max=32)

    def _accuracy(net, data_loader):
        net.train(False)

        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda())
            X.requires_grad = True

            score = net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        net.train(True)
        return 100 * num_correct / num_total

    best_acc = 0.0
    best_epoch = None
    for t in range(100):
        epoch_loss = []
        num_correct = 0
        num_total = 0
        cnt = 0
        print('Epoch ' + str(t), flush=True)
        for X, y in train_loader:
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda())
            solver.zero_grad()

            X.requires_grad = True
            score = net(X)
            loss = criterion(score, y)
            epoch_loss.append(loss.data.item())

            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data)

            loss.backward()
            solver.step()


            if (num_total >= cnt * 2000):
                cnt += 1
                logger.info("Train Acc: " + str((100 * num_correct / num_total).item()) + "%" + "\n" + str(
                    num_correct) + " " + str(num_total) + "\n" + str(prediction) + " " + str(y.data) + "\n" + str(
                    loss.data))
                logger.handlers[1].flush()

        train_acc = (100 * num_correct / num_total).item()
        test_acc = _accuracy(net, test_loader)
        lrscheduler.step(test_acc)
        # lrscheduler.step(sum(epoch_loss) / len(epoch_loss))
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = t + 1
            print('*', end='',flush=True)
        logger.info('%d\t%.10f\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
              (t + 1, solver.param_groups[0]['lr'],sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        logger.handlers[1].flush()

    torch.save(net.module.state_dict(),'./data/GCN_STN_BLN_DiffLr.pth')



if __name__ == '__main__':
    main()