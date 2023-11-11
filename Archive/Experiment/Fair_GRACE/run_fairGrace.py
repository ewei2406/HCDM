# %%
import argparse
from model import Grace
from aug import aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn

from eval import label_classification, eval_unbiasedness_movielens
import warnings

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='movielens')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=str, default='random')
parser.add_argument('--debias_method', type=str, default='uge-w', choices=['uge-r', 'uge-w', 'uge-c', 'none'], help='debiasing method to apply')
parser.add_argument('--debias_attr', type=str, default='age', help='sensitive attribute to be debiased')
parser.add_argument('--reg_weight', type=float, default=0.2, help='weight for the regularization based debiasing term')  

parser.add_argument('--epochs', type=int, default=100, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')

parser.add_argument('--act_fn', type=str, default='relu')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--seed", type=int, default=100, help='seed')
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
debias_method = args.debias_method

# Step 2: Prepare data =================================================================== #
if debias_method in ['uge-w', 'uge-c']:
    dataset = '{}_debias_{}'.format(args.dataname, args.debias_attr)
else:
    dataset = args.dataname

graph = load(dataset)
in_dim = graph.ndata['feat'].shape[1]

# %%
from dataset import SENSITIVE_ATTR_DICT  # predefined sensitive attributes for different datasets
from dataset import DATA_FOLDER
import pandas as pd

# Group nodes
debias_attr = args.debias_attr
attribute_list = SENSITIVE_ATTR_DICT[args.dataname]

non_sens_attr_ls = [attr for attr in attribute_list if attr!=debias_attr]
non_sens_attr_idx = [i for i in range(len(attribute_list)) if attribute_list[i]!=debias_attr]

attribute_file = '{}/{}_node_attribute.csv'.format(DATA_FOLDER, args.dataname)
node_attributes = pd.read_csv(attribute_file)

attr_comb_groups = node_attributes.groupby(attribute_list)
nobias_comb_groups = node_attributes.groupby(non_sens_attr_ls)

attr_comb_groups_map = {tuple(group[1].iloc[0]):list(group[1].index) 
                        for group in attr_comb_groups}
nobias_attr_comb_groups_map = {tuple(group[1].iloc[0][non_sens_attr_ls]):list(group[1].index) 
                            for group in nobias_comb_groups}

print ('Group finished.')
print ('  attr_comb_group_num:', len(attr_comb_groups_map.keys()))
print ('  nobias_attr_comb_group_num:', len(nobias_attr_comb_groups_map.keys()))

# %%
def map_tuple(x, index_ls):
  return tuple([x[idx] for idx in index_ls])

def mem_eff_matmul_mean(mtx1, mtx2):
  mtx1_rows = list(mtx1.shape)[0]
  if mtx1_rows <= 1000:
    return th.mean(th.matmul(mtx1, mtx2))
  else:
    value_sum = 0
    for i in range(mtx1_rows // 1000):
      value_sum += th.sum(th.matmul(mtx1[i*1000:(i+1)*1000, :], mtx2))
    if mtx1_rows % 1000 != 0:
      value_sum += th.sum(th.matmul(mtx1[(i+1)*1000:, :], mtx2))
    return value_sum / (list(mtx1.shape)[0] * list(mtx2.shape)[1])

# %%
import random

# Step 3: Create model =================================================================== #
model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
model = model.to(args.device)
print(f'# params: {count_parameters(model)}')

optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# Step 4: Training =======================================================================
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    graph1, feat1 = aug(graph, graph.ndata['feat'], drop_feature_rate_1, drop_edge_rate_1)
    graph2, feat2 = aug(graph, graph.ndata['feat'], drop_feature_rate_2, drop_edge_rate_2)

    graph1 = graph1.to(args.device)
    graph2 = graph2.to(args.device)

    feat1 = feat1.to(args.device)
    feat2 = feat2.to(args.device)

    loss = model(graph1, graph2, feat1, feat2)
    
    # UGE-R
    if debias_method in ['uge-r', 'uge-c']:
        h1 = model.encoder(graph1, feat1)
        h2 = model.encoder(graph2, feat2)
        regu_loss = 0
        scr_groups = random.sample(list(attr_comb_groups_map.keys()), 100)  
        dst_groups = random.sample(list(attr_comb_groups_map.keys()), 100)
        nobias_scr_groups = [map_tuple(group, non_sens_attr_idx) for group in scr_groups]
        nobias_dst_groups = [map_tuple(group, non_sens_attr_idx) for group in dst_groups]

        for group_idx in range(len(scr_groups)):
            for view in [h1, h2]:
                scr_group_nodes = attr_comb_groups_map[scr_groups[group_idx]]
                dsc_group_nodes = attr_comb_groups_map[dst_groups[group_idx]]
                
                scr_node_embs = view[scr_group_nodes]
                dsc_node_embs = view[dsc_group_nodes]
                aver_score = mem_eff_matmul_mean(scr_node_embs, dsc_node_embs.T)

                nobias_scr_group_nodes = nobias_attr_comb_groups_map[nobias_scr_groups[group_idx]]
                nobias_dsc_group_nodes = nobias_attr_comb_groups_map[nobias_dst_groups[group_idx]]
                nobias_scr_node_embs = view[nobias_scr_group_nodes]
                nobias_dsc_node_embs = view[nobias_dsc_group_nodes]
                nobias_aver_score = mem_eff_matmul_mean(nobias_scr_node_embs, nobias_dsc_node_embs.T)

                regu_loss += th.square(aver_score - nobias_aver_score)
            
        print(f"Epoch={epoch:03d}, loss: {loss.item():.2f}, regu_loss: {regu_loss.item():.2f}")

        loss += args.reg_weight * regu_loss / 100
    else:
        print(f"Epoch={epoch:03d}, loss: {loss.item():.2f}")

    loss.backward()
    optimizer.step()

    # print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

# Step 5: Linear evaluation ============================================================== #
print("=== Final ===")

graph = graph.add_self_loop()
graph = graph.to(args.device)
embeds = model.get_embedding(graph, graph.ndata['feat'].to(args.device))

# %%
from eval import label_classification, eval_unbiasedness_movielens
'''Evaluation Embeddings  '''
# label_classification(embeds, graph.ndata['label'], graph.ndata['train_mask'], graph.ndata['test_mask'], split=args.split)
res = eval_unbiasedness_movielens('movie', embeds.cpu())

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import Archive.Utils.Export as Export

results = {
  "dataname": args.dataname,
  "epochs": args.epochs,
  "debias_method": args.debias_method,
  "debias_attr": args.debias_attr,
  "reg_weight": args.reg_weight,
  "temp": args.temp,
  "der1": args.der1,
  "der2": args.der2,
  "dfr1": args.dfr1,
  "dfr2": args.dfr2,
  "gender_f1m": res['unbiasedness']['gender'],
  "age_f1m": res['unbiasedness']['age'],
  "occupation_f1m": res['unbiasedness']['occupation'],
  "link_ndcg": res['utility'],
}

Export.saveData('./results.csv', results)