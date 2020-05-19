# crop size 448
# mean 109.973, 127.336, 123.883
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

'''

'''
class CUB200_loader(data.Dataset):
    def __init__(self, root, split='train', transform=None):

        std = 1. / 255.
        means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]

        # std = 255.
        # means = [109.97 , 127.34 , 123.88 ]

        # split_list = open(os.path.join(root, 'train_test_split.txt')).readlines()
        split_list = open(os.path.join(root, 'tts2.txt')).readlines()
        self.idx2name = []
        classes = open(os.path.join(root, 'classes.txt')).readlines()
        self._imgpath = []
        self._imglabel = []
        self.transform = transform

        for c in classes:
            idx, name = c.strip().split()
            self.idx2name.append(name)

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

        train_list = []
        test_list = []
        for line in split_list:
            idx, is_train = line.strip().split()
            if int(is_train) == 1:
                train_list.append(int(idx) - 1)
            else:
                test_list.append(int(idx) - 1)

        image_list = open(os.path.join(root, 'images.txt')).readlines()
        image_list = np.array(image_list)
        label_list = open(os.path.join(root, 'image_class_labels.txt')).readlines()
        label_list = np.array(label_list)

        if split.lower() == 'train':
            train_images = image_list[train_list]
            train_labels = label_list[train_list]
            for i, fname in enumerate(train_images):
                idx, path = fname.strip().split()
                self._imgpath.append(os.path.join(root, 'images', path))
                idx, label = train_labels[i].strip().split()
                self._imglabel.append(int(label) - 1)
        else:
            test_images = image_list[test_list]
            test_labels = label_list[test_list]
            for i, fname in enumerate(test_images):
                idx, path = fname.strip().split()
                self._imgpath.append(os.path.join(root, 'images', path))
                idx, label = test_labels[i].strip().split()
                self._imglabel.append(int(label) - 1)

    def __getitem__(self, index):
        img = cv2.imread(self._imgpath[index])
        # try:
        img = self.transform(img)
        # tmp=torch.max(img)
        cls = self._imglabel[index]
        return img, cls
        # except:
        #     print(self._imgpath[index],flush=True)
        #     print(self.transform,flush=True)
        #     print(index,flush=True)
        #     return None,None

    def __len__(self):
        return len(self._imglabel)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def CUB_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls


if __name__ == '__main__':
    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
    trainloader = data.DataLoader(trainset, batch_size=32,
                                  shuffle=False, collate_fn=trainset.CUB_collate, num_workers=1)

    cnt=0
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        print('pixel\' value max', torch.max(img),torch.min(img))
        if cnt > 100:
            break

    testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011')
    testloader = data.DataLoader(testset, batch_size=32,
                                 shuffle=False, collate_fn=testset.CUB_collate, num_workers=1)

    for img, cls in trainloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break
