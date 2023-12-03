import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import json

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""


class TrafficDataset(Dataset):
    def __init__(self, root, filename, test=True, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.root = root
        self.test = test
        self.filename = filename
        super(TrafficDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return 'not_implemented.pt'

    def download(self):
        pass

    def process(self):
        f = open(self.root + '/raw/' + self.filename, 'r')
        for index, line in enumerate(f):
            json_file = self.root + '/raw/jsongraphs/' + line[:-1]
            temp = open(json_file)
            jsondata = json.load(temp)

            # Get node features
            node_feats = self._get_node_features(jsondata['objects'])
            # Get edge features
            edge_feats = self._get_edge_features(jsondata['edges'])
            # Get adjacency info
            edge_index = self._get_adjacency_info(jsondata['edges'])


            # Get labels info
            label = self._get_labels(index)

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_{index}.pt'))

    def _get_node_features(self, data_nodes):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        CLASS_DICT = {'pedestrian':0, 'person':0, 'bike':1, 'motorcycle':1, 'car':2, 'truck':3, 'bus':3}
        LENGTH_DICT = {'pedestrian':0.5/7, 'person':0.5/7, 'bike':1/7, 'motorcycle':1/7, 'car':4/7, 'truck':7/7, 'bus':7/7}
        WIDTH_DICT = {'pedestrian':0.5/2.5, 'person':0.5/2.5, 'bike':0.5/2.5, 'motorcycle':0.5/2.5, 'car':2/2.5, 'truck':2.5/2.5, 'bus':2.5/2.5}
        MAX_VEL = 25    # m/s (~ 90km/h)
        MAX_LENGTH = 7 # metres
        MAX_WIDTH = 2.5   # metres
        num_nodes = len(data_nodes)

        all_node_feats = []

        for i in range(num_nodes):
            node_feats = [0]*4
            if data_nodes[i]['classification'] not in CLASS_DICT:
                data_nodes[i]['classification'] = 'car'
            node_feats[CLASS_DICT[data_nodes[i]['classification']]] = 1
            node_feats.append(float(data_nodes[i]['vel']) / MAX_VEL)
            node_feats.append(float(data_nodes[i]['vx']) / MAX_VEL)
            node_feats.append(float(data_nodes[i]['vy']) / MAX_VEL)
            node_feats.append(LENGTH_DICT[data_nodes[i]['classification']])
            node_feats.append(WIDTH_DICT[data_nodes[i]['classification']])

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, data_edges):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        EDGE_DICT = {}
        MAX_DISTANCE = 60   # metres
        num_edges = len(data_edges)

        all_edge_feats = []

        for i in range(num_edges):
            if (data_edges[i]['head'], data_edges[i]['tail']) not in EDGE_DICT:
                EDGE_DICT[(data_edges[i]['head'], data_edges[i]['tail'])] = len(all_edge_feats)
                all_edge_feats.append([0]*6)
            edge_idx = EDGE_DICT[(data_edges[i]['head'], data_edges[i]['tail'])]

            if data_edges[i]['type'] == 'longitudinal':
                all_edge_feats[edge_idx][0], all_edge_feats[edge_idx][1] = 1, 1-float(data_edges[i]['path_distance'])/MAX_DISTANCE

            elif data_edges[i]['type'] == 'lateral':
                all_edge_feats[edge_idx][2], all_edge_feats[edge_idx][3] = 1, 1-float(data_edges[i]['path_distance'])/MAX_DISTANCE

            elif data_edges[i]['type'] == 'intersecting':
                all_edge_feats[edge_idx][4], all_edge_feats[edge_idx][5] = 1, 1-float(data_edges[i]['path_distance'])/MAX_DISTANCE

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, data_edges):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        num_edges = len(data_edges)

        edge_indices = []

        for i in range(num_edges):
            if (data_edges[i]['head'], data_edges[i]['tail']) not in edge_indices:            
                edge_indices += [[data_edges[i]['head'], data_edges[i]['tail']]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, index):
        label = np.load(self.root + '/raw/labels.npy')[index]
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        with open(self.root + '/raw/' + self.filename, 'r') as fp:
            return len(fp.readlines())

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data