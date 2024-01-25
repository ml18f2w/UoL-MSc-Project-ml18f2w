import os
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from stl import mesh
import pandas as pd
import math

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/Aneurisk/models")


class Aneurisk(Dataset):
    '''
    process Aneurisk data and generate dataloader
    '''
    def __init__(self, train_mode='train', cls_state=True, npoints=1024, data_aug=True):
        self.npoints = npoints  # 2048 pts
        self.data_augmentation = data_aug
        self.data = []
        self.labels = []
        self.cls_state = cls_state
        self.train_mode = train_mode

        # get xyz from stl files and get label from csv files
        for i in range(1, 100):
            if i < 10:
                path = BASE + "/C000" + str(i)
            else:
                path = BASE + "/C00" + str(i)

            if os.path.exists(path):
                stl_path = path + "/surface/model.stl"
                csv_path = path + "/manifest.csv"
                stl_file = mesh.Mesh.from_file(stl_path)
                csv_file = pd.read_csv(csv_path, sep=',')
                label = (csv_file['ruptureStatus'].values == 'U')[0]
                self.labels.append(label)
                points = stl_file.points
                points[:, 0] = (points[:, 0] + points[:, 3] + points[:, 6]) / 3
                points[:, 1] = (points[:, 1] + points[:, 4] + points[:, 7]) / 3
                points[:, 2] = (points[:, 2] + points[:, 5] + points[:, 8]) / 3
                xyz = points[:, :3]
                if xyz.shape[0] < self.npoints:
                    choice = np.random.choice(xyz.shape[0], self.npoints, replace=True)
                else:
                    choice = np.random.choice(xyz.shape[0], self.npoints, replace=False)
                xyz = xyz[choice, :]
                self.data.append(xyz)

        # scatter and slice data
        shuffled_id = np.arange(len(self.data))
        np.random.seed(42)
        np.random.shuffle(shuffled_id)
        train_id = math.ceil(len(self.data) * 8 / 10)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.data = self.data[shuffled_id]
        self.labels = self.labels[shuffled_id]
        if self.train_mode == 'train':
            self.data = self.data[:train_id, :, :]
            self.labels = self.labels[:train_id]
        elif self.train_mode == 'test':
            self.data = self.data[train_id:, :, :]
            self.labels = self.labels[train_id:]
        else:
            print("Error")
            raise Exception("training mode invalid")

    def __getitem__(self, index):
        point_set = self.data[index]
        cls = torch.from_numpy(np.array(self.labels[index]).astype(np.int64))
        # normalization to unit ball
        point_set = point_set - np.mean(point_set, axis=0)  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        # data augmentation
        if self.data_augmentation:
            if self.train_mode == 'train':
                point_set = random_scale(point_set)
                point_set = translate_pointcloud(point_set)
            if self.train_mode == 'test':
                # point_set[:, :3] = random_scale(point_set[:, :3])
                # point_set[:, :3] = translate_pointcloud(point_set[:, :3])
                point_set = point_set

        point_set = torch.from_numpy(point_set)
        return point_set, cls

    def __len__(self):
        return self.data.shape[0]


def get_train_valid_loader(num_workers=0, pin_memory=False, batch_size=4, npoints=2048):
    '''
    get train data and test data
    '''
    train_dataset = Aneurisk(train_mode='train', cls_state=True, npoints=npoints, data_aug=True)
    valid_dataset = Aneurisk(train_mode='test', cls_state=True, npoints=npoints, data_aug=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader, len(train_dataset)


def random_scale(point_data, scale_low=0.8, scale_high=1.2):
    """
    Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    scaled_pointcloud = np.multiply(point_data, scale).astype('float32')
    return scaled_pointcloud


def translate_pointcloud(pointcloud):
    '''
    panning image data
    '''
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(pointcloud, shift).astype('float32')
    return translated_pointcloud
