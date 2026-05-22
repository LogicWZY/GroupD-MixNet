import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class UltrasoundDataset(data.Dataset):
    """
    dataloader for Ultrasound segmentation tasks
    """
    def __init__(self, dataset, train_file_dir, resize, augmentations):
        
        if dataset == "BUSI": # Dataset/BUSI/BUSI_train_1.txt
            with open(train_file_dir, "r") as f1:
                img_list = f1.readlines()
            self.images = ["Dataset/"+dataset+"/image/"+item.replace("\n", "") +".png" for item in img_list]
            self.gts = ["Dataset/"+dataset+"/gt/"+item.replace("\n", "") +"_mask.png" for item in img_list]
        else:
            assert f"No Dataset!  Current Dataset : {dataset}"

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if augmentations == 'True':
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((resize, resize)),
                transforms.ToTensor()])
            
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        # print(self.images[index])
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(dataset, train_file_dir, resize, batchsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = UltrasoundDataset(dataset, train_file_dir, resize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, dataset, test_file_dir, testsize):
        self.testsize = testsize
        
        if dataset == "BUSI": # Dataset/BUSI/BUSI_train_1.txt
            with open(test_file_dir, "r") as f1:
                img_list = f1.readlines()
            self.images = ["Dataset/"+dataset+"/image/"+item.replace("\n", "") +".png" for item in img_list]
            self.gts = ["Dataset/"+dataset+"/gt/"+item.replace("\n", "") +"_mask.png" for item in img_list]
        
        else:
            assert f"No Dataset!  Current Dataset : {dataset}"
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # print(self.images[self.index])
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def get_list(self):
        return self.images, self.gts


if __name__ == "__main__":
    dataset = UltrasoundDataset(dataset="BUSI", train_file_dir="Dataset/BUSI/BUSI_train_4.txt",resize=224, augmentations=True)
    train_data = get_loader(dataset="BUSI", train_file_dir="Dataset/BUSI/BUSI_train_4.txt",resize=224,batchsize=16)
    val = test_dataset(dataset="BUSI", test_file_dir="Dataset/BUSI/BUSI_test_4.txt",  testsize=224)
    val.load_data()