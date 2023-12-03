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
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
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
        f = open('./data/raw/' + self.filename, 'r')
        for index, line in enumerate(f):
            json_file = './data/raw/jsongraphs/' + line[:-1]
            temp = open(json_file)
            data = json.load(temp)

            # Get node features
            node_feats = self._get_node_features(data['objects'])
            # Get edge features
            edge_feats = self._get_edge_features(data['edges'])
            # Get adjacency info
            edge_index = self._get_edge_features(data['edges'])

            # Get labels info
            label = self._get_labels()

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
        CLASS_DICT = {'pedestrian':0, 'bike':1, 'car':2, 'truck':3}
        MAX_VEL = 30    # m/s (~ 110km/h)
        MAX_LENGTH = 20 # metres
        MAX_WIDTH = 3   # metres
        num_nodes = len(data_nodes)

        all_node_feats = []

        for i in range(num_nodes):
            node_feats = [0]*4
            node_feats[CLASS_DICT[data_nodes[i]['classification']]] = 1
            node_feats.append(float(data_nodes[i]['vel']) / MAX_VEL)
            node_feats.append(float(data_nodes[i]['vx']) / MAX_VEL)
            node_feats.append(float(data_nodes[i]['vy']) / MAX_VEL)
            node_feats.append(float(data_nodes[i]['length']) / MAX_LENGTH)
            node_feats.append(float(data_nodes[i]['width']) / MAX_WIDTH)

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
        num_edges = len(data_edges)

        all_edge_feats = []

        for i in range(num_edges):
            edge_feats = []

            if (data_edges[i]['head'], data_edges[i]['tail']) not in EDGE_DICT:
                EDGE_DICT[(data_edges[i]['head'], data_edges[i]['tail'])] = len(edge_feats)
                edge_feats.append([0]*6)
            edge_idx = EDGE_DICT[(data_edges[i]['head'], data_edges[i]['tail'])]

            if data_edges[i]['type'] == 'longitudinal':
                edge_feats[edge_idx][0], edge_feats[edge_idx][1] = 1, float(data_edges[i]['path_distance'])

            elif data_edges[i]['type'] == 'lateral':
                edge_feats[edge_idx][2], edge_feats[edge_idx][3] = 1, float(data_edges[i]['path_distance'])

            elif data_edges[i]['type'] == 'intersecting':
                edge_feats[edge_idx][4], edge_feats[edge_idx][5] = 1, float(data_edges[i]['path_distance'])

            # Append node features to matrix (once, per direction)
            all_edge_feats += [edge_feats]

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
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self):
        label = np.load('./data/raw/labels.npy')
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

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
    

dataset = TrafficDataset('data', 'jsonfiles.txt')