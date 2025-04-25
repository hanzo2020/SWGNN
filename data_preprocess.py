import os
import argparse
import pandas as pd
import numpy as np
import torch
import time
import pickle
import torch as t
import torch_geometric
from torch_geometric.utils import k_hop_subgraph, degree
torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import config_load
from tqdm import tqdm
from utils import *

EPS = 1e-8

class CancerDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CancerDataset, self).__init__('.', None, None)
        self.data_list = data_list
        self._data, self.slices = self.collate(self.data_list)
        self.num_slices = len(self.data_list)
        self.num_nodes = self.data_list[0].num_nodes
        self._data.num_classes = 2

    def get_idx_split(self, i):
        train_idx = torch.where(self.data_list[0].train_mask[:, i])[0]#这相当于只取了最后一列，不知道啥情况
        test_idx = torch.where(self.data_list[0].test_mask[:, i])[0]
        valid_idx = torch.where(self.data_list[0].valid_mask[:, i])[0]

        return {
            'train': train_idx,
            'test': test_idx,
            'valid': valid_idx
        }

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    

def create_cv_dataset(train_idx_list, valid_idx_list, test_idx_list, data_list):#数据能从这里生成？
    num_nodes = data_list[0].num_nodes
    num_folds = len(train_idx_list)

    train_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    valid_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    test_mask = np.zeros((num_nodes, num_folds), dtype=bool)
    for i in range(num_folds):
        train_mask[train_idx_list[i], i] = True
        valid_mask[valid_idx_list[i], i] = True
        test_mask[test_idx_list[i], i] = True

    for data in data_list:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.unlabeled_mask = ~torch.logical_or(
            data.train_mask[:, 0], torch.logical_or(data.valid_mask[:, 0], data.test_mask[:, 0]))
    cv_dataset = CancerDataset(data_list=data_list)

    return cv_dataset


def get_degrees(dataset):
    data_list = []
    for data in dataset.data_list:
        degrees = t.bincount(data.edge_index.flatten()) / 2
        num_nodes = degrees.size(0)
        adjacency_list = {i: set() for i in range(num_nodes)}
        for start, end in tqdm(data.edge_index.T, desc='load graph degrees'):
            adjacency_list[start.item()].add(end.item())
        sum_of_degrees = np.zeros(num_nodes, dtype=int)
        for node in range(num_nodes):
            two_hop_list = list({two_hop_neighbor for neighbor in adjacency_list[node] for two_hop_neighbor in adjacency_list[neighbor]})
            adjacency_list[node].add(node)
            two_hop_list = [x for x in two_hop_list if x not in adjacency_list[node]]
            sum_of_degrees[node] = len(two_hop_list)
        degrees_f = degrees.unsqueeze(1).to(torch.float32)
        ratio = torch.where(degrees != 0, torch.tensor(sum_of_degrees) / degrees,  torch.zeros_like(degrees))
        ratio = ratio.unsqueeze(1).to(torch.float32)
        data.degrees = degrees_f
        data.adj_degrees = torch.tensor(sum_of_degrees).unsqueeze(1).to(torch.float32)
        data.ratio = ratio
        data_list.append(data)
    dataset_DSGNN = CancerDataset(data_list=data_list)
    return dataset_DSGNN

def get_all_edge(dataset):
    edge_list_all = []
    for data in dataset.data_list:
        edge_list_all.append(data.edge_index)
    edge_index = torch.cat(edge_list_all, dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32)
    data_list = []
    for data in dataset.data_list:
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data_list.append(data)
    dataset_all_edge = CancerDataset(data_list=data_list)
    return dataset_all_edge




def get_data(configs, stable=True):
    data_dir = configs["data_dir"]
    load_data = configs["load_data"]

    cell_line = get_cell_line(data_dir)

    def get_dataset_dir(stable):
        dataset_suffix = "_dataset_final" if stable else "_dataset"
        dataset_dir = os.path.join(
            data_dir, cell_line + dataset_suffix + '.pkl')
        
        return dataset_dir


    if load_data:
        dataset_dir = get_dataset_dir(stable)
        print(f"Loading dataset from: {dataset_dir} ......")
        with open(dataset_dir, 'rb') as f:
            cv_dataset = pickle.load(f)
        if configs['model'] == 'DSGNN':
            print('DSGNN dataset loader')
            start_time = time.time()
            cv_dataset = get_degrees(cv_dataset)
            end_time = time.time()
            print(f"运行时间: {end_time - start_time:.4f} 秒")
        return cv_dataset
    
