# %%
import torch as th
import numpy as np
import pandas as pd
import dgl

# %%
# import json
# raw_f = json.load(open('./processed_data/LastFM_node_feature.json'))
# # 7624 nodes, 7842 features
# feat = th.zeros([7624, 7842])

# for i in raw_f.keys():
#   feat[int(i), th.tensor(raw_f[i], dtype=th.long)] = 1

# np.savetxt("LastFM_node_feature.csv", feat.numpy(), delimiter=",", fmt='%d')

# %%
# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='LastFM')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--debias_method', type=str, default='uge-r', choices=['uge-r', 'none', 'random'], help='debiasing method to apply')
parser.add_argument('--debias_attr', type=int, default=0, help='idx of sensitive attribute to be debiased')
parser.add_argument('--num_sens', type=int, default=3, help='# of sensitive attr to make')
parser.add_argument('--reg_weight', type=float, default=0.2, help='weight for the regularization based debiasing term')  

parser.add_argument('--epochs', type=int, default=200, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--seed", type=int, default=100, help='seed')
parser.add_argument('--der1', type=float, default=0.2, help='Drop edge ratio of the 1st augmentation.')

parser.add_argument('--sim_diff_ratio', type=float, default=5, help='Drop feature ratio of the 2nd augmentation.')
parser.add_argument('--enable_heuristic', type=str, default='Y', help='Drop feature ratio of the 2nd augmentation.')


args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

np.random.seed(args.seed)
th.manual_seed(args.seed)

if args.device != 'cpu':
    th.cuda.manual_seed(args.seed)

# %%
def load_graph_from_file(edge_file: str, feat_file: str, label_file:str=None, disable_header=False) -> dgl.DGLGraph:
  edges = pd.read_csv(edge_file, engine='c')
  edges = th.tensor(edges.to_numpy()).t()
  graph = dgl.graph((edges[0], edges[1]))
  
  if disable_header:
    feat = pd.read_csv(feat_file, header=None, engine='c')
  else:
    feat = pd.read_csv(feat_file, engine='c')

  feat = th.tensor(feat.to_numpy()).int()
  graph.ndata['feat'] = feat

  if label_file:
    labels = pd.read_csv(label_file, engine='c')
    labels = th.tensor(labels.to_numpy())
    graph.ndata['labels'] = labels.t()[1]

  return graph


def add_sens_(graph: dgl.DGLGraph, indices: th.tensor):
  sens = graph.ndata['feat'][:, indices].clone()
  inverse = th.full((graph.num_nodes(),), 1)
  inverse[indices] = 0
  graph.ndata['feat'][indices] = 0
  graph.ndata['sens_attr'] = sens


def calc_weights_(graph: dgl.DGLGraph, debias_attr: int, ratio: int, target: int):
  sens = graph.ndata['sens_attr'].t()[debias_attr]
  sim = th.tensor([sens[edge[0]] == sens[edge[1]] for edge in graph.adj().coalesce().indices().t()]).int()
  sim = sim * (ratio - 1)
  sim = sim + 1
  sim = sim * target * sim.shape[0] / sim.sum()
  graph.edata['weight'] = sim.clamp(0,1)


# %%
graph = load_graph_from_file(
  edge_file=f'./processed_data/{args.dataname}_edge.csv', 
  feat_file=f'./processed_data/{args.dataname}_node_feature.csv', 
  disable_header=True)

add_sens_(graph, graph.ndata['feat'].sum(dim=0).topk(args.num_sens).indices)

if args.enable_heuristic:
  calc_weights_(graph, args.debias_attr, ratio=args.sim_diff_ratio, target=args.der1)
else:
  calc_weights_(graph, args.debias_attr, ratio=1, target=args.der1)

# %%
node_attributes = pd.DataFrame(graph.ndata['sens_attr'].numpy())
attribute_list = list(node_attributes.columns)
non_sens_attr_ls = [i for i in attribute_list if i!=args.debias_attr]
non_sens_attr_idx = [i for i in range(len(attribute_list)) if attribute_list[i]!=args.debias_attr]

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
def aug_weight(graph: dgl.DGLGraph, drop_feat: float, drop_edge: float=0.2):
  edge_mask = th.bernoulli(graph.edata['weight']) == 0
  masked_edges = graph.adj().coalesce().indices()[:, edge_mask]

  new_graph = dgl.graph((masked_edges[0], masked_edges[1])).to(graph.device)

  feat_mask = th.rand((graph.ndata['feat'].shape[1])) < (drop_feat)
  new_graph.ndata['feat'] = graph.ndata['feat'].clone()

  new_graph.ndata['feat'][:, feat_mask] = 0
  new_graph = new_graph.add_self_loop()
  return new_graph

def aug(graph: dgl.DGLGraph, drop_feat: float, drop_edge: float=0.2):
  edge_mask = th.bernoulli(th.full((graph.num_edges(),), drop_edge)) == 0
  masked_edges = graph.adj().coalesce().indices()[:, edge_mask]

  new_graph = dgl.graph((masked_edges[0], masked_edges[1])).to(graph.device)

  feat_mask = th.rand((graph.ndata['feat'].shape[1])) < (drop_feat)
  new_graph.ndata['feat'] = graph.ndata['feat'].clone()

  new_graph.ndata['feat'][:, feat_mask] = 0
  new_graph = new_graph.add_self_loop()
  return new_graph


aug_weight(graph, 0.2)

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
import importlib
import model
importlib.reload(model)
import random

import torch.nn as nn

dr = 0.2
# Step 3: Create emb_model =================================================================== #
emb_model = model.Grace(
  in_dim=graph.ndata['feat'].shape[1], 
  hid_dim=args.hid_dim, 
  out_dim=args.out_dim, 
  num_layers=args.num_layers, 
  act_fn=nn.ReLU(), 
  temp=args.temp
)
emb_model = emb_model.to(args.device)

def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
print(f'# params: {count_parameters(emb_model)}')

optimizer = th.optim.Adam(emb_model.parameters(), lr=args.lr, weight_decay=args.wd)

# Step 4: Training =======================================================================
for epoch in range(args.epochs):
    emb_model.train()
    optimizer.zero_grad()

    view_1 = aug_weight(graph, drop_feat=dr).to(args.device)
    view_2 = aug_weight(graph, drop_feat=dr).to(args.device)

    loss = emb_model(view_1, view_2, view_1.ndata['feat'], view_1.ndata['feat'], batch_size=0)
    
    # UGE-R
    if args.debias_method in ['uge-r', 'uge-c']:
        h1 = emb_model.encoder(view_1, view_1.ndata['feat'])
        h2 = emb_model.encoder(view_2, view_2.ndata['feat'])
        regu_loss = 0
        scr_groups = random.sample(list(attr_comb_groups_map.keys()), 8)  
        dst_groups = random.sample(list(attr_comb_groups_map.keys()), 8)
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
            
        # print(f"Epoch={epoch:03d}, loss: {loss.item():.2f}, regu_loss: {regu_loss.item():.2f}")

        loss += args.reg_weight * regu_loss / 100
    
    loss.backward()
    optimizer.step()

    # print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

# Step 5: Linear evaluation ============================================================== #
graph = graph.add_self_loop()
graph = graph.to(args.device)
embeds = emb_model.get_embedding(graph, graph.ndata['feat'].to(args.device)).cpu()

# %%
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def get_f1(embeds, graph, debias_attr):
  evaluate_attr = graph.ndata['sens_attr'][:,debias_attr]
  split_idx = int(graph.num_nodes() * 0.75)
  lgreg = LogisticRegression(
    random_state=0, 
    class_weight='balanced', 
    max_iter=500).fit(
    embeds[:split_idx].cpu(), evaluate_attr[:split_idx].cpu())
  pred = lgreg.predict(embeds[split_idx:].cpu())

  score = f1_score(evaluate_attr[split_idx:split_idx + pred.shape[0]].cpu(), pred, average='micro')

  print(f'-- micro-f1 when predicting sensitive attr #{debias_attr}: {score}')
  return score
  

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg


def eval_link_ndcg(embeds, graph: dgl.DGLGraph):
  accum_ndcg = 0
  node_cnt = 0
  sample_size = int(min(graph.num_nodes() / 25, 75))
  k = int(min(sample_size / 5, 10))
  adj = graph.adj().to_dense()

  for node in graph.nodes():
    node_edges = adj[node]
    positive_nodes = node_edges.nonzero(as_tuple=True)[0]
    split_idx = int(positive_nodes.shape[0] / 10) + 1

    if split_idx == 0 or split_idx > sample_size:
      continue
      
    negative_nodes = np.random.choice(
      (1 - node_edges).nonzero(as_tuple=True)[0], 
      sample_size - split_idx,
      replace=False
    )
    positive_nodes = positive_nodes[:split_idx] # Subset of positive pair
    eval_nodes = np.concatenate((positive_nodes, negative_nodes))
    eval_edges = np.zeros(sample_size)
    eval_edges[:split_idx] = 1

    predicted_edges = np.dot(embeds[node], embeds[eval_nodes].T)
    rank_pred_keys = np.argsort(predicted_edges)[::-1]
    ranked_node_edges = eval_edges[rank_pred_keys]
    ndcg = ndcg_at_k(ranked_node_edges, k)
    accum_ndcg += ndcg

    node_cnt += 1

  score = 1 - (accum_ndcg/node_cnt)
  print(f'-- ndcg of link prediction: {score:.6f}')
  return score

# %%
if args.debias_method == 'random':
  embeds = th.rand(embeds.shape)

# %%
results = {
  "dataname": args.dataname,
  "epochs": args.epochs,
  "seed": args.seed,
  "debias_method": args.debias_method,
  "debias_attr": args.debias_attr,
  "reg_weight": args.reg_weight,
  "temp": args.temp,
  "der1": args.der1,
  "enable_heuristic": args.enable_heuristic,
  "ratio": args.sim_diff_ratio
}

for attr_idx in range(0, graph.ndata['sens_attr'].shape[1]):
  results[f'f1_{attr_idx}'] = get_f1(embeds, graph, attr_idx)
results['link'] = eval_link_ndcg(embeds.cpu(), graph.cpu())

# %%
import sys
import os
sys.path.append(os.path.join('../..'))
import Archive.Utils.Export as Export
Export.saveData('./results_H.csv', results)


