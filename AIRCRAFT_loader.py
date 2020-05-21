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
class AIR100_loader(data.Dataset):
    def __init__(self, root, split='train', transform=None):

        root=root.replace('\\', '/')
        # split_list = open(os.path.join(root, 'train_test_split.txt')).readlines()
        # split_list = open(os.path.join(root, 'tts2.txt')).readlines()

        self.hash2id = {}
        self.id2hash = []
        nameOrder=open(os.path.join(root, 'images_box.txt')).readlines()
        for i , line in enumerate(nameOrder):
            hh=line.strip().split()[0]
            self.hash2id[hh]=i
            self.id2hash.append(hh)

        self.idx2name = []
        self.name2idx = {}
        classes = open(os.path.join(root, 'variants.txt')).readlines()
        for i, c in enumerate(classes):
            name = c.strip()
            self.idx2name.append(name)
            self.name2idx[name]=i

        self._imgpath = []
        self._imglabel = []

        if split.lower() == 'train':
            image_label_list=open(os.path.join(root, 'images_variant_trainval.txt')).readlines()
        else:
            image_label_list = open(os.path.join(root, 'images_variant_test.txt')).readlines()

        for line in image_label_list:
            self._imgpath.append(os.path.join(root, 'images', line.strip().split()[0]+'.jpg'))
            self._imglabel.append(self.name2idx[' '.join(line.strip().split()[1:])])

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


        # for line in split_list:
        #     idx, is_train = line.strip().split()
        #     if int(is_train) == 1:
        #         train_list.append(int(idx) - 1)#1 to 0
        #     else:
        #         test_list.append(int(idx) - 1)

        # image_list = open(os.path.join(root, 'images.txt')).readlines()
        # image_list = np.array(image_list)
        # label_list = open(os.path.join(root, 'image_class_labels.txt')).readlines()
        # label_list = np.array(label_list)
        #
        # if split.lower() == 'train':
        #     train_images = image_list[train_list]
        #     train_labels = label_list[train_list]
        #     for i, fname in enumerate(train_images):
        #         idx, path = fname.strip().split()
        #         self._imgpath.append(os.path.join(root, 'images', path))
        #         idx, label = train_labels[i].strip().split()
        #         self._imglabel.append(int(label) - 1)
        # else:
        #     test_images = image_list[test_list]
        #     test_labels = label_list[test_list]
        #     for i, fname in enumerate(test_images):
        #         idx, path = fname.strip().split()
        #         self._imgpath.append(os.path.join(root, 'images', path))
        #         idx, label = test_labels[i].strip().split()
        #         self._imglabel.append(int(label) - 1)

    def __getitem__(self, index):
        img = cv2.imread(self._imgpath[index])
        img = self.transform(img)
        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imglabel)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def AIR_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls


if __name__ == '__main__':
    trainset = AIR100_loader(os.getcwd() + '/data/AIRCRAFT')
    trainloader = data.DataLoader(trainset, batch_size=6,
                                  shuffle=False, collate_fn=trainset.AIR_collate, num_workers=1)

    cnt=0
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        print('pixel\' value max', torch.max(img),torch.min(img))
        cnt+=1
        if cnt > 10:
            break

    testset = AIR100_loader(os.getcwd() + '/data/AIRCRAFT')
    testloader = data.DataLoader(testset, batch_size=6,
                                 shuffle=False, collate_fn=testset.AIR_collate, num_workers=1)

    for img, cls in trainloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break
