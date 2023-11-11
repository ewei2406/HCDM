from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset



"""
Make our own datasets
We store three components for each dataset
-- node_feature.csv: store node feature
-- node_label.csv: store node label
-- edge.csv: store the edges
"""

import torch
from dgl.data import DGLDataset
from dgl import backend as F
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import os.path
import dgl


RAW_FOLDER = './raw_data'
DATA_FOLDER = './processed_data'
WEIGHT_FOLDER = './precomputed_weights'

SENSITIVE_ATTR_DICT = {
    'movielens': ['gender', 'occupation', 'age'],
    'pokec': ['gender', 'region', 'AGE'],
    'pokec-z': ['gender', 'region', 'AGE'],
    'pokec-n': ['gender', 'region', 'AGE'],
}


class MyDataset(DGLDataset):
    def __init__(self, data_name="pokec-n_debias_gender", data_folder=DATA_FOLDER, weight_folder=WEIGHT_FOLDER, raw_folder=RAW_FOLDER):
        self.data_name = data_name
        self.data_folder = data_folder
        self.weight_folder = weight_folder
        self.raw_folder = raw_folder
        super().__init__(name='customized_dataset')

    def process(self):
        raw_folder = self.raw_folder
        processed_folder = self.data_folder
        weight_folder = self.weight_folder
        
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        
        # !Key place to triger UGE-W
        # - load edge (biased or reweighted) based on data_name
        # - if data_name includes "debias", it means we are loading precomputed edge weights for uge-w
        # - otherwise, we are loading original 0/1 edges
        if 'debias' in self.data_name:  # e.g. self.data_name==movielens_debias_gender to trigger uge-w
            edge_file = '{}/{}_edge.csv'.format(weight_folder, self.data_name)
            print('Precomputed weights for weighting-based debiasing UGE-W Loaded')
            
        else:  # e.g. self.data_name==movielens without triggering uge-w
            edge_file = '{}/{}_edge.csv'.format(processed_folder, self.data_name)
        
        node_feat_file = '{}/{}_node_feature.csv'.format(processed_folder, self.data_name.split('_')[0])
        node_label_file = '{}/{}_node_label.csv'.format(processed_folder, self.data_name.split('_')[0])
        node_attribute_file = '{}/{}_node_attribute.csv'.format(processed_folder, self.data_name.split('_')[0])  # sensitive node attributes predefined to debias
        
        ### download raw data and process into unified csv format ###
        # if self.data_name.split('_')[0] == 'movielens' and not os.path.exists('{}/ml-1m/users.dat'.format(RAW_FOLDER)):
        #     process_raw_movielens(raw_folder, processed_folder)
        # elif self.data_name.split('_')[0].startswith('pokec') and not os.path.exists('{}/pokec/region_job.csv'.format(RAW_FOLDER)):
        #     process_raw_pokec(raw_folder, processed_folder, self.data_name.split('_')[0])
        
        ### create dgl graph from customized data ###
        
        print ('Creating DGL graph...')
        
        # Load the data as DataFrame
        edges = pd.read_csv(edge_file, engine='python')
        node_features = pd.read_csv(node_feat_file, engine='python')
        node_labels = pd.read_csv(node_label_file, engine='python')
        node_attributes = pd.read_csv(node_attribute_file, engine='python')
        
        c = node_labels['Label'].astype('category')
        classes = dict(enumerate(c.cat.categories))
        self.num_classes = len(classes)

        # Transform from DataFrame to torch tensor
        node_features = torch.from_numpy(node_features.to_numpy()).float()
        node_labels = torch.from_numpy(node_labels['Label'].to_numpy()).long()
        edge_features = torch.from_numpy(edges['Weight'].to_numpy()).float()
        edges_src = torch.from_numpy(edges['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges['Dst'].to_numpy())

        # construct DGL graph
        g = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        # !Key place to triger UGE-W or not by data_name
        g.edata['weight'] = edge_features if 'debias' in self.data_name else torch.ones_like(edge_features)
        
        
        # add sensitive attribute information to graph
        for l in list(node_attributes):
            g.ndata[l] = torch.from_numpy(node_attributes[l].to_numpy()).long()
                
        # rewrite the to_bidirected function to support edge weights on bidirected graph (aggregated)
        # self.graph = dgl.to_bidirected(g)
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = dgl.to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)
        
        # zero in-degree nodes will lead to invalid output value
        # a common practice to avoid this is to add a self-loop
        self.graph = dgl.add_self_loop(g)

        # For node classification task, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        # ! We currently only target on link prediction task
        # ! this is a placeholder for node classification task
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.graph.ndata['feat'].shape[1]))
        # print('  NumClasses: {}'.format(self.num_classes))
        # print('  NumTrainingSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['train_mask']).shape[0]))
        # print('  NumValidationSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['val_mask']).shape[0]))
        # print('  NumTestSamples: {}'.format(
        #     F.nonzero_1d(self.graph.ndata['test_mask']).shape[0]))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'pokec':
        dataset = MyDataset('pokec-n_debias_gender')
    elif name == 'movielens':
        dataset = MyDataset('movielens')
    else:
        dataset = MyDataset(name)

    graph = dataset[0]

    # train_mask = graph.ndata.pop('train_mask')
    # test_mask = graph.ndata.pop('test_mask')

    # feat = graph.ndata.pop('feat')
    # labels = graph.ndata.pop('label')

    return graph


if __name__ == "__main__":
    a = MyDataset()