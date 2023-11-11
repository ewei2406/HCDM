# %% [markdown]
# # Diffusion Model test

# %%
import torch_geometric
import dgl
import random
import torch
from torch.utils.data import DataLoader
import selection
import argparse
import schedules
import numpy as np

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--subgraph_size', type=int, default=32)
parser.add_argument('--num_save', type=int, default=50)
parser.add_argument('--file', type=str, default="./out.csv")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--timesteps', type=int, default=400)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--method', type=str, default='walk', choices=[
    'walk', 'rand', 'cluster'
])
parser.add_argument('--dataset', type=str, default='cora', choices=[
    'cora', 'citeseer'
])

args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu': torch.cuda.manual_seed(args.seed)

# %%
if args.dataset == 'cora': dataset = dgl.data.CoraGraphDataset()

g = dataset[0]
subgraph_size = args.subgraph_size
batch_size = args.batch_size
timesteps = args.timesteps

if args.method == 'walk': method = selection.subgraph_node2vec_random_walk
if args.method == 'rand': method = selection.subgraph_random
if args.method == 'cluster': method = selection.subgraph_cluster

# %%
dataset = []
channels = 1
for i in range(batch_size * 50):
    dataset.append({"pixel_values" : (method(subgraph_size, g).adj().to_dense().unsqueeze(0) * 2) - 1})

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
from torch.optim import Adam
from unet import Unet
import torch

model = Unet(
    dim=subgraph_size,
    channels=channels,
    dim_mults=(1, 2, 4, 8)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

# %%
import importlib
import unet
importlib.reload(unet)

unet.trainUnet(
    model=model,
    epochs=args.epochs,
    dataloader=dataloader,
    optimizer=Adam(model.parameters(), lr=1e-3),
    device=device,
    timesteps=timesteps,
    scheduler=schedules.linear_beta_schedule
)

# %%
import sampling
samples = sampling.sample(
    model, 
    image_size=subgraph_size, 
    batch_size=batch_size, 
    channels=channels, 
    scheduler=schedules.linear_beta_schedule, 
    timesteps=timesteps
)

# %%
import matplotlib.pyplot as plt

def showPics(num, imgs):
    plt.figure()
    f, axarr = plt.subplots(1, num, figsize=(15,15)) 
    for i, img in enumerate(imgs):
        # print(i['pixel_values'][0].squeeze().shape)
        axarr[i].imshow(img)
        axarr[i].axis('off')
        if i == num - 1: break



def mirror(A: np.ndarray):
    return np.tril(A) + np.triu(A.T, 1)

train_data = [x["pixel_values"].squeeze() for x in dataset]
pred_data = [mirror(x) for x in (samples[-1].squeeze() > 0)]

comparisons = 10
showPics(comparisons, train_data)
showPics(comparisons, pred_data)

# %%
import networkx as nx

def calculateStats(adj: torch.tensor, label: str, index: int):
    out = {}
    adj = mirror(adj > 0)
    nxg = nx.from_numpy_matrix(adj)

    out['label'] = label
    out['index'] = index
    out['num_edges'] = adj.sum()
    out['num_nodes'] = adj.shape[0]
    out['density'] = adj.sum() / adj.shape[0]
    out['num_triangles'] = int(sum(nx.triangles(nxg).values()) / 3)
    out['max_diameter'] = max([max(j.values()) for (i,j) in nx.shortest_path_length(nxg)])

    out.update(vars(args))
    out.pop("file")
    return out


# %%
calculateStats(pred_data[0], "pred", i)

# %%
calculateStats(train_data[0].numpy(), "train", i)

# %%
import export

for i, idx in enumerate(random.sample(range(len(train_data)), args.num_save)):
    export.saveData(args.file, calculateStats(train_data[idx].numpy(), "train", i))

for i, idx in enumerate(random.sample(range(len(pred_data)), args.num_save)):
    export.saveData(args.file, calculateStats(pred_data[idx], "pred", i))


