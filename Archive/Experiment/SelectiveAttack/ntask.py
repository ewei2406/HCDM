import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from Dataset import Dataset
# from AAGNN.loadGraph import loadGraph, loadPokec
# from AAGNN.GCN import GCN
# from AAGNN import metrics
# from AAGNN import utils

import dataloader
import torch
import numpy as np
import argparse

################################################
# Configuration
################################################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')

parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--protect_size', type=float, default=0.1, help='Number of randomly chosen protected nodes')
parser.add_argument('--random_select', type=str, default="N", help='Choose a class instead')
parser.add_argument('--ptb_rate', type=float, default=0.5, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--reg_epochs', type=int, default=75, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=15, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=10, help='Epochs to train surrogate before perturb')

parser.add_argument('--csv', type=str, default='', help='save the outputs to csv')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--ntasks', type=int, default=1, help='number of additional tasks')
args = parser.parse_args()

################################################
# Environment
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

print('==== Environment ====')
print(f'  torch version: {torch.__version__}')
print(f'  device: {device}')
print(f'  torch seed: {args.seed}')

################################################
# Load data
################################################

print(f'==== Dataset: {args.dataset} ====')

if args.dataset == 'pokec':
    adj, labels, features, idx_train, idx_val, idx_test = loadPokec(device)
else:
    adj, labels, features, idx_train, idx_val, idx_test = loadGraph('./datasets', args.dataset, 'gcn', args.seed, device)

tasks = {0: labels}
for task in range(args.ntasks):
    tasks[task + 1], features = utils.get_task(0, features)

total_acc = { "baseline": {}, "locked": {} }

print(tasks)

################################################
# Protected set
################################################

if args.random_select == "Y":
    g0 = torch.rand(features.shape[0]) <= args.protect_size
else:
    g0 = labels == 1

g_g0 = ~g0

print(f"Number of protected nodes: {g0.sum():.0f}")
print(f"Protected Size: {g0.sum() / features.shape[0]:.2%}")

################################################
# Baseline model
################################################

print(f'==== Training baselines ====')

for t in tasks:
    print(tasks[t])

    baseline = GCN(
        input_features=features.shape[1],
        output_classes=tasks[t].max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"B T{t}"
    )

    baseline.fit(
        features=features, 
        adj=adj, 
        labels=tasks[t], 
        idx_train=idx_train, 
        idx_test=idx_test, 
        epochs=args.reg_epochs
    )

    pred = baseline(features, adj)
    base_acc = metrics.partial_acc(pred, tasks[t], g0, g_g0)
    total_acc["baseline"][t] = base_acc

################################################
# Perturbing
################################################

surrogates = {}

for t in tasks:
    surrogates[t] = GCN(
        input_features=features.shape[1],
        output_classes=tasks[t].max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"S T{t}"
    )

    surrogates[t].fit(
        features=features, 
        adj=adj, 
        labels=tasks[t], 
        idx_train=idx_train, 
        idx_test=idx_test, 
        epochs=args.surrogate_epochs
    )

perturbations = torch.zeros_like(adj).float()
perturbations.requires_grad = True
num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

t = tqdm(range(args.ptb_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
t.set_description("Perturbing")

def loss_func(predictions, labels):
    loss = 0
    p = predictions[0]
    loss += F.cross_entropy(predictions[p][g0], labels[p][g0])
    loss -= F.cross_entropy(predictions[p][g_g0], labels[p][g_g0])

    for p in predictions:
        # loss += F.cross_entropy(predictions[p][g0], labels[p][g0])
        loss -= F.cross_entropy(predictions[p], labels[p])

    return loss

for epoch in t:
    # Perturb

    modified_adj = utils.get_modified_adj(adj, perturbations)

    predictions = {}
    for s in surrogates:
        surrogates[s].eval()
        predictions[s] = surrogates[s](features, modified_adj)

    loss = loss_func(predictions, tasks)

    adj_grad = torch.autograd.grad(loss, perturbations)[0]

    lr = (num_perturbations * 2.5 * (args.ntasks + 1)) / (adj_grad.abs().sum())
            
    perturbations = perturbations + (lr * adj_grad)

    pre_projection = int(perturbations.sum() / 2)

    perturbations = utils.projection(perturbations, num_perturbations)

    # Train

    total_loss = 0
    for s in surrogates:
        modified_adj = utils.get_modified_adj(adj, perturbations)
        total_loss += surrogates[s].train1epoch(features, modified_adj, tasks[s], idx_train, idx_test)

    t.set_postfix({
        "adj_l": loss.item(),
        "adj_g": int(adj_grad.sum()),
        "pre-p": pre_projection,
        "target": int(num_perturbations / 2),
        "loss": total_loss
    })

with torch.no_grad():

    max_loss = -1000

    for k in range(0,3):
        sample = torch.bernoulli(perturbations)
        modified_adj = utils.get_modified_adj(adj, perturbations)
        modified_adj = utils.make_symmetric(modified_adj) # Removing this creates "impossible" adj, but works well

        predictions = {}
        for s in surrogates:
            surrogates[s].eval()
            predictions[s] = surrogates[s](features, modified_adj)

        loss = loss_func(predictions, tasks)

        if loss > max_loss:
            max_loss = loss
            best = sample
    
    print(f"Best sample loss: {loss:.2f}\t Edges: {best.abs().sum() / 2:.0f}")

################################################
# Train on locked graph
################################################

print(f'==== Training locked ====')

locked_adj = utils.get_modified_adj(adj, best)

for t in tasks:
    print(tasks[t])

    locked = GCN(
        input_features=features.shape[1],
        output_classes=tasks[t].max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"L T{t}"
    )

    locked.fit(
        features=features, 
        adj=locked_adj, 
        labels=tasks[t], 
        idx_train=idx_train, 
        idx_test=idx_test, 
        epochs=args.reg_epochs
    )

    pred = locked(features, locked_adj)
    lock_acc = metrics.partial_acc(pred, tasks[t], g0, g_g0)
    total_acc["locked"][t] = lock_acc

################################################
# Evaluation
################################################

locked_adj = utils.get_modified_adj(adj, best)
change = locked_adj - adj
change = change.to(device)

print(f"Randomly selected: {args.random_select}\n")
print(f"Ptb rate: {args.ptb_rate}\n")

print("==== Accuracies ====")
print(f"          ΔG0\tΔGX")
for t in tasks:
    dg0 = total_acc["locked"][t]["g0"] - total_acc["baseline"][t]["g0"]
    dgX = total_acc["locked"][t]["gX"] - total_acc["baseline"][t]["gX"]
    print(f"task {t} | {dg0:.1%}\t{dgX:.1%}")

print()
print("=====EDGE CHANGES=====")
metrics.show_metrics(change, tasks[0], g0, device)

