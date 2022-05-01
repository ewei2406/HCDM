import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

############################
#region Arguments

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

parser.add_argument('--ptb_rate', type=float, default=0.25, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--protect_size', type=float, default=0.1, help='Number of randomly chosen protected nodes')
parser.add_argument('--reg_epochs', type=int, default=100, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=30, help='Epochs to perturb adj matrix')

parser.add_argument('--save', type=str, default='N', help='save the outputs to csv')

args = parser.parse_args()

#endregion
##########################

##########################
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
##########################

########################
#region Data

from Utils import GraphData

print(f'[i] Dataset: {args.dataset}')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)
graph.summarize()

#endregion
########################

########################
#region Feature selection

print(graph.features)

#endregion
########################




