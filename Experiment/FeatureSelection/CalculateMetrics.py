# %%
import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), '../..'))

# %%
########################
#region Arguments

import argparse

parser = argparse.ArgumentParser()

# data args
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

# model args
parser.add_argument('--reg_epochs', type=int, default=100, help='Epochs to train models')
parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

args = parser.parse_args()

#endregion
########################

########################
#region Environment

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

print(f'[i] Environment:  torch {torch.__version__} \tdevice: {device} \tseed: {args.seed}')

#endregion
########################

########################
#region Data

from Utils import GraphData

print(f'[i] Dataset: {args.dataset}')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)
graph.summarize()

#endregion
########################

# %%
########################
#region Feature Selection

from Utils import FeatureMetrics
from Utils import Utils
from Utils import Export
from tqdm import tqdm

print(graph)
#endregion
########################