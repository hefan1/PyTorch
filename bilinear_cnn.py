# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.
Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_all.py --base_lr 0.05 \
        --batch_size 64 --epochs 100 --weight_decay 5e-4
"""


import os

import torch
import torchvision
import torch.utils.data as data

from CUB_loader import CUB200_loader
from AIRCRAFT_loader import AIR100_loader
from CARS_loader import CARS196_loader

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)




import logging
def get_log(file_name):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler() # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever



    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger

# logger=get_log('/data/oriBcnn.log')
logger=get_log('test.log')


class BCNN(torch.nn.Module):
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
        torch.nn.init.kaiming_normal(self.fc.weight.data)  # 何凯明初始化
        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)  # fc层的bias进行constant初始化

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
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(size=448),  # Let smaller edge match
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(size=448),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                  std=(0.229, 0.224, 0.225))
])
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.CenterCrop(size=448),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                  std=(0.229, 0.224, 0.225))
])

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
        self._path = path['dataset']
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        #no cuda
        # self._net=BCNN()
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
        # trainset = CUB200_loader(self._path,transform=train_transforms)
        # testset = CUB200_loader(self._path,split='test',transform=test_transforms)
        # self._train_loader = data.DataLoader(trainset, batch_size=self._options['batch_size'],
        #                               shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)#shuffle?
        # self._test_loader = data.DataLoader(testset, batch_size=self._options['batch_size'],
        #                              shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)


    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
            cnt = 0
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                #no cuda
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda())

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

                if(num_total>=cnt*500):
                    cnt=cnt+1
                    logger.info("Train Acc: "+str((100 * num_correct / num_total).item())+"%"
                                +'\n'+str(num_correct)+" "+str(num_total)
                                +'\n'+str(prediction)+" "+str(y.data)
                                +'\n'+str(loss.data))


                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = (100 * num_correct / num_total).item()
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
            logger.info('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        torch.save(self._net.state_dict(),'/data/bilinearOri.pth')
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
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda())

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total


    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        # train_data = cub200.CUB200(
        #     root=self._path['cub200'], train=True,
        #     transform=torchvision.transforms.ToTensor(), download=True)
        # train_loader = torch.utils.data.DataLoader(
        #     train_data, batch_size=1, shuffle=False, num_workers=4,
        #     pin_memory=True)
        # trainset = AIR100_loader(self._path, transform=train_transforms)
        # trainset = CARS196_loader(self._path, transform=train_transforms)
        trainset = CUB200_loader(self._path,transform=train_transforms)
        train_loader = data.DataLoader(trainset, batch_size=4,
                                      shuffle=False, collate_fn=trainset.CUB_collate, num_workers=4)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(trainset))
        std.div_(len(trainset))
        print(mean,flush=True)
        print(std,flush=True)


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
        'dataset': os.path.join(project_root, 'data/CUB_200_2011'),# 'dataset': os.path.join(project_root, 'data/AIRCRAFT'),
        'model': os.path.join(project_root, 'model', args.model),
    }
    #not now
    # for d in path:
    #     if d == 'model':
    #         assert os.path.isfile(path[d])
    #     else:
    #         assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    manager.getStat()
    # manager.train()


if __name__ == '__main__':
    main()