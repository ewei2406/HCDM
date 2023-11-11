import argparse
from model import Grace
from aug import aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn

from eval import label_classification
import warnings

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=str, default='random')
parser.add_argument('--one_view', type=str, default='N')
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--epochs', type=int, default=200, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')

parser.add_argument('--act_fn', type=str, default='relu')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--der1', type=float, default=0.2, help='Drop edge ratio of the 1st augmentation.')
parser.add_argument('--der2', type=float, default=0.2, help='Drop edge ratio of the 2nd augmentation.')
parser.add_argument('--dfr1', type=float, default=0.2, help='Drop feature ratio of the 1st augmentation.')
parser.add_argument('--dfr2', type=float, default=0.2, help='Drop feature ratio of the 2nd augmentation.')

args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

np.random.seed(args.seed)
th.manual_seed(args.seed)

if args.device != 'cpu':
    th.cuda.manual_seed(args.seed)

if __name__ == '__main__':

    # Step 1: Load hyperparameters =================================================================== #
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph = load(args.dataname)
    in_dim = graph.ndata['feat'].shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f'# params: {count_parameters(model)}')

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, graph.ndata['feat'], drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, graph.ndata['feat'], drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        if args.one_view == 'Y':
            loss = model(graph.to(args.device), graph2, graph.ndata['feat'].to(args.device), feat2)
        else:
            loss = model(graph1, graph2, feat1, feat2)

        loss.backward()
        optimizer.step()

        # print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")

    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    embeds = model.get_embedding(graph, graph.ndata['feat'].to(args.device))

    '''Evaluation Embeddings  '''
    acc = label_classification(embeds, graph.ndata['label'], graph.ndata['train_mask'], graph.ndata['test_mask'], split=args.split)

    import sys
    import os
    sys.path.append(os.path.join('../..'))
    import Archive.Utils.Export as Export

    results = {
        "dataname": args.dataname,
        "epochs": args.epochs,
        "temp": args.temp,
        "der1": args.der1,
        "der2": args.der2,
        "dfr1": args.dfr1,
        "dfr2": args.dfr2,
        "f1_micro": acc['F1Mi']['mean'],
        "f1_macro": acc['F1Ma']['mean'],
        "one_view": args.one_view,
    }

    Export.saveData('./resultsOneView.csv', results)


