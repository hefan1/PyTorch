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


from CUB_loader import CUB200_loader


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class BCNN_STN(torch.nn.Module):
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
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 200)


        self.c_size=512

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1*self.c_size, 8*self.c_size, kernel_size=7),  # 22
            nn.MaxPool2d(2, stride=2),  # 11
            nn.ReLU(True),
            nn.Conv2d(8*self.c_size, 10*self.c_size, kernel_size=5),  # 7
            nn.MaxPool2d(2, stride=2),  # 7/2=3
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*self.c_size * 3 * 3, 16*self.c_size),
            nn.ReLU(True),
            nn.Linear(16*self.c_size, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        #Transform failed
        # X=self.stn(X)


        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        # print(X.shape)
        # for i in range(N):
        #     for j in range(512):
        #         X[i][j]=self.stn(torch.reshape(X[i][j],(1,-1,28,28)))

        X=self.stn(X)

        X = X.view(N, 512, 28**2)
        # print(X.shape)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X

class BCNNManager(object):
    """Manager class to train bilinear CNN.
    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        # Network.
        #self._net = torch.nn.DataParallel(BCNN()).cuda()
        #no cuda
        self._net=BCNN_STN()
        # Load the model from disk.
        #self._net.load_state_dict(torch.load(self._path['model']))#not now
        print(self._net)
        # Criterion.
        #self._criterion = torch.nn.CrossEntropyLoss().cuda()
        self._criterion = torch.nn.CrossEntropyLoss()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        # train_data = cub200.CUB200(
        #     root=self._path['cub200'], train=True, download=True,
        #     transform=train_transforms)
        # test_data = cub200.CUB200(
        #     root=self._path['cub200'], train=False, download=True,
        #     transform=test_transforms)
        # self._train_loader = torch.utils.data.DataLoader(
        #     train_data, batch_size=self._options['batch_size'],
        #     shuffle=True, num_workers=4, pin_memory=True)
        # self._test_loader = torch.utils.data.DataLoader(
        #     test_data, batch_size=16,
        #     shuffle=False, num_workers=4, pin_memory=True)
        trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
        testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
        self._train_loader = data.DataLoader(trainset, batch_size=self._options['batch_size'],
                                      shuffle=True, collate_fn=trainset.CUB_collate, num_workers=1)#shuffle?
        self._test_loader = data.DataLoader(testset, batch_size=self._options['batch_size'],
                                     shuffle=False, collate_fn=testset.CUB_collate, num_workers=1)


    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                #no cuda
                # X = torch.autograd.Variable(X.cuda())
                # y = torch.autograd.Variable(y.cuda(async=True))

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                # epoch_loss.append(loss.data[0])
                epoch_loss.append(loss.data.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)

                if(num_total%1==0):
                    print("Train Acc: "+str((100 * num_correct / num_total).item())+"%")
                    print(str(num_correct)+" "+str(num_total))
                    print(str(prediction)+" "+str(y.data))
                    print(str(loss.data))

                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.
        Args:
            data_loader: Train/Test DataLoader.
        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            # X = torch.autograd.Variable(X.cuda())
            # y = torch.autograd.Variable(y.cuda(async=True))

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase


    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        # train_data = cub200.CUB200(
        #     root=self._path['cub200'], train=True,
        #     transform=torchvision.transforms.ToTensor(), download=True)
        # train_loader = torch.utils.data.DataLoader(
        #     train_data, batch_size=1, shuffle=False, num_workers=4,
        #     pin_memory=True)
        trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
        train_loader = data.DataLoader(trainset, batch_size=4,
                                      shuffle=False, collate_fn=trainset.CUB_collate, num_workers=1)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(trainset))
        std.div_(len(trainset))
        print(mean)
        print(std)


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model for fine-tuning.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': os.path.join(project_root, 'data/CUB_200_2011'),
        'model': os.path.join(project_root, 'model', args.model),
    }
    #not now
    # for d in path:
    #     if d == 'model':
    #         assert os.path.isfile(path[d])
    #     else:
    #         assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    #manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()