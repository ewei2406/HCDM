import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


########################
#region Arguments

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')


parser.add_argument('--reg_epochs', type=int, default=50, help='epochs')

parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--load_ptb', type=str, default='', help='temp var to load')

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

print('==== Environment ====')
print(f'  torch version: {torch.__version__}')
print(f'  device: {device}')
print(f'  torch seed: {args.seed}')

#endregion
########################

########################
#region Load data

from Archive.Utils import GraphData, FeatureMetrics

print(f'==== Dataset: {args.dataset} ====')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)
graph.summarize()

#endregion
########################

########################
#region Load perturbations

from Archive.Utils import Export, Metrics, Utils

imported_data = Export.load_var(f'pertubations-{args.dataset}@0.5', './perturbations.json')
best = torch.tensor(imported_data[0], device=device)
g0 = torch.tensor(imported_data[1], device=device)
gX = ~g0

#endregion
########################

########################
#region Test

from tqdm import tqdm
from Archive.Models.GCN import GCN


t = tqdm(range(0, graph.features.shape[1]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
t.set_description("Metrics calculating")
locked_adj = Utils.get_modified_adj(graph.adj, best)

features_t = graph.features.t().contiguous()

for feat_idx in t:
    labels = FeatureMetrics.discretize(features_t[feat_idx]).long().t()
    if labels.max().item() == 0:
        continue
    features = torch.cat((graph.features[:, 0:feat_idx], graph.features[:, feat_idx+1:]), dim=1)
    baseline_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"baseline"
    ).to(device)

    baseline_model.fitManual(features, graph.adj, labels, graph.idx_train, graph.idx_test, args.reg_epochs, verbose=False)
    pred = baseline_model(features, graph.adj)
    baseline_acc = Metrics.partial_acc(pred, labels, g0, gX, verbose=False)

    locked_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"locked"
    ).to(device)

    locked_model.fitManual(features, locked_adj, labels, graph.idx_train, graph.idx_test, args.reg_epochs, verbose=False)
    pred = locked_model(features, locked_adj)
    locked_acc = Metrics.partial_acc(pred, labels, g0, gX, verbose=False)
    
    results = {
        "feat_idx": feat_idx,
        "dataset": args.dataset,
        "reg_epochs": args.reg_epochs,
        "ptb_rate": 0.5,
        "base_g0": baseline_acc['g0'],
        "base_gX": baseline_acc['gX'],
        "lock_g0": locked_acc['g0'],
        "lock_gX": locked_acc['gX'],
    }

    Export.saveData(f"featureResults-{args.dataset}.csv", results)

#endregion
########################

