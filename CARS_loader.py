# crop size 448
# mean 109.973, 127.336, 123.883
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.io import loadmat


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

'''

'''
class CARS196_loader(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        # print(os.path.join(root, 'data','CAR_196','car_devkit','devkit','cars_train_annos.mat'))
        self.idx2name = []
        self.name2idx = {}
        classes=loadmat(os.path.join(root, 'car_devkit','devkit','cars_meta.mat'))['class_names'][0]
        for i, c in enumerate(classes):
            name = str(c[0]).strip()
            self.idx2name.append(name)
            self.name2idx[name] = i


        self._imgpath = []
        self._imglabel = []
        imgBase=''

        if split.lower() == 'train':
            image_label_list=loadmat(os.path.join(root, 'car_devkit','devkit','cars_train_annos.mat'))['annotations'][0]
            imgBase=os.path.join(root, 'cars_train')
        else:
            image_label_list =loadmat(os.path.join(root, 'car_devkit','devkit','cars_test_annos_withlabels.mat'))['annotations'][0]
            imgBase = os.path.join(root, 'cars_test')

        for line in image_label_list:
            self._imgpath.append(os.path.join(imgBase, str(line['fname'][0])))
            self._imglabel.append(int(str(line['class'][0][0])))

        self.transform = transform

        if transform is None and split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.RandomCrop([224, 224]),
                transforms.RandomHorizontalFlip(),#flip?
                transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))#toTensor had normalization
            ])
        elif transform is None and split.lower() == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))#,transforms.Normalize(mean=means,std=[std] * 3)
            ])
        else:
            print(" [*] Warning: transform is not None, Recomend to use defualt")
            pass


    def __getitem__(self, index):
        img = cv2.imread(self._imgpath[index])
        img = self.transform(img)
        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imglabel)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def CARS_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls


if __name__ == '__main__':
    trainset = CARS196_loader(os.path.join(os.getcwd(),'data','CAR_196'))
    trainloader = data.DataLoader(trainset, batch_size=6,
                                  shuffle=False, collate_fn=trainset.CARS_collate, num_workers=1)

    cnt=0
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        print('pixel\' value max', torch.max(img),torch.min(img))
        cnt+=1
        if cnt > 10:
            break

    testset = CARS196_loader(os.path.join(os.getcwd(),'data','CAR_196'),split='test')
    testloader = data.DataLoader(testset, batch_size=6,
                                 shuffle=False, collate_fn=testset.CARS_collate, num_workers=1)

    for img, cls in trainloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break
