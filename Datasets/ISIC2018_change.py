import os
from glob import glob

import PIL
import torch
import numpy as np
import cv2

from os import listdir
from os.path import join
from PIL import Image
from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ISIC2018_dataset_change(Dataset):
    def __init__(self, dataset_folder='mnt/code/data/ISIC2018_Task1_npy_all',
                 folder='folder1', train_type='train', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.folder_file = './Datasets/' + folder
        self.dataset_folder = dataset_folder

        if self.train_type in ['train']:
            with open(join(self.dataset_folder, 'train.list'), 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]

        elif self.train_type in ['validation']:
            with open(join(self.dataset_folder, 'val.list'), 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]

        elif self.train_type in ['test']:
            print(self.dataset_folder, 'val.list')
            with open(join(self.dataset_folder, 'val.list'), 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

    def __getitem__(self, item: int):
        # image = cv2.imread(self.image_list[item])
        # label = cv2.imread(self.mask_list[item], 0)

        image = np.load(self.folder[item])
        label = np.load(self.mask[item])

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)

        if self.train_type in ['validation']:
            name = self.folder[item].split('\\')[-1]
            return name, sample['image'], sample['label']
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.image_list)

# a = ISIC2018_dataset()
