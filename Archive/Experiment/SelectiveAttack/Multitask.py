# %%
import sys
import os
sys.path.append(os.path.join('../..'))

################################################
# Arguments
################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

parser.add_argument('--method', type=str, default='SLL', help='method')
parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--protect_size', type=float, default=0.1, help='Number of randomly chosen protected nodes')
parser.add_argument('--ptb_rate', type=float, default=0.25, help='Perturbation rate (percentage of available edges)')

# simplistic1, simplistic2, noise, SLLnoSample, SLL
parser.add_argument('--numtasks', type=int, default=3, help='num additional tasks')
parser.add_argument('--sample_size', type=int, default=500, help='')
parser.add_argument('--num_samples', type=int, default=20, help='')


parser.add_argument('--reg_epochs', type=int, default=100, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=30, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=0, help='Epochs to train surrogate before perturb')

parser.add_argument('--save', type=str, default='N', help='save the outputs to csv')
parser.add_argument('--save_location', type=str, default="./SelectiveAttack.csv", help='where to save the outputs to csv')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')


parser.add_argument('--check_universal', type=str, default='N', help='check universal protection')

args = parser.parse_args()

################################################
# Environment
################################################

import torch
import numpy as np

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

print('==== Environment ====')
print(f'  torch version: {torch.__version__}')
print(f'  device: {device}')
print(f'  torch seed: {args.seed}')

################################################
# Dataset
################################################

from Archive.Utils import GraphData

print(f'==== Dataset: {args.dataset} ====')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)
graph.summarize()

tasks = graph.features.sum(dim=0).topk(k=args.numtasks).indices
print(tasks)

################################################
# Designate protected
################################################

g0 = torch.rand(graph.features.shape[0]) <= args.protect_size
# g0 = graph.labels == 5 
g0 = g0.to(device)
gX = ~g0

print(f"Number of protected nodes: {g0.sum():.0f}")
print(f"Protected Size: {g0.sum() / graph.features.shape[0]:.2%}")

# %%
################################################
# Sampling Matrix
################################################

from SamplingMatrix import SamplingMatrix

samplingMatrix = SamplingMatrix(g0, gX, graph.adj, args.sample_size)

samplingMatrix.get_sample()
samplingMatrix.getRatio()

# %%
a = torch.tensor([1, 2, 3])
a.long()

# %%
# %%
################################################
# Generate Perturbations (RANDOM NOISE)
################################################
from Archive.Utils import Utils as Utils
import torch.nn.functional as F
from Archive.Models.GCN import GCN

surrogate = GCN(
      input_features=graph.features.shape[1],
      output_classes=graph.labels.max().item()+1,
      hidden_layers=args.hidden_layers,
      device=device,
      lr=args.model_lr,
      dropout=args.dropout,
      weight_decay=args.weight_decay,
      name=f"surrogate"
  ).to(device)

surrogates = {}
for task in tasks:
  m = graph.features[:,task].max().item() + 1
  m = int(m)
  print(m)
  surrogates[task.item()] = GCN(
      input_features=graph.features.shape[1],
      output_classes=m,
      hidden_layers=args.hidden_layers,
      device=device,
      lr=args.model_lr,
      dropout=args.dropout,
      weight_decay=args.weight_decay,
      name=f"surrogate{task}"
  ).to(device)

import torch.nn.functional as F
from tqdm import tqdm

perturbations = torch.zeros_like(graph.adj).float()
count = torch.zeros_like(graph.adj).float()
num_perturbations = args.ptb_rate * graph.adj.sum()

t = tqdm(range(args.ptb_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
t.set_description("Perturbing")

for epoch in t:
  # Re-initialize adj_grad
  adj_grad = torch.zeros_like(graph.adj).float()

  # Get modified adj
  modified_adj = Utils.get_modified_adj(graph.adj, perturbations).float().to(device)

  if args.method == 'SLL':

    for sample_epoch in range(args.num_samples):
      # Get sample indices
      # sampled = torch.bernoulli(sampling_matrix)
      idx = samplingMatrix.get_sample()
      # print(idx)

      # Map sample to adj
      sample = modified_adj[idx[0], idx[1]].clone().detach().requires_grad_(True).to(device)
      modified_adj[idx[0], idx[1]] = sample

      # Get grad
      # predictions = surrogate(graph.features, modified_adj)

      loss = 0
      for s in surrogates:
        surrogates[s].eval()
        modified_feat = graph.features.clone()
        modified_feat[:, s] = 0
        pred = surrogates[s](modified_feat, modified_adj)

        loss += F.cross_entropy(pred[g0], graph.features[g0, s].long()) \
          - F.cross_entropy(pred[gX], graph.features[gX, s].long())

      predictions = surrogate(graph.features, modified_adj)
      loss += F.cross_entropy(predictions[g0], graph.labels[g0]) \
          - F.cross_entropy(predictions[gX], graph.labels[gX])

      grad = torch.autograd.grad(loss, sample)[0]

      # Implement averaging
      adj_grad[idx[0], idx[1]] += grad
      count[idx[0], idx[1]] += 1

      # Update the sampling matrix
      samplingMatrix.updateByGrad(adj_grad, count)
      samplingMatrix.getRatio()

      # Average the gradient
      adj_grad = torch.div(adj_grad, count)
      adj_grad[adj_grad != adj_grad] = 0
  
  else:
    # Get grad
    modified_adj = modified_adj.clone().detach().requires_grad_(True).to(device)

    loss = 0
    for s in surrogates:
      surrogates[s].eval()
      modified_feat = graph.features.clone()
      modified_feat[:, s] = 0
      pred = surrogates[s](modified_feat, modified_adj)

      loss += F.cross_entropy(pred[g0].long(), graph.features[g0, s]) \
        - F.cross_entropy(pred[gX], graph.features[gX, s])

    predictions = surrogate(graph.features, modified_adj)
    loss += F.cross_entropy(predictions[g0], graph.labels[g0]) \
        - F.cross_entropy(predictions[gX], graph.labels[gX])

    grad = torch.autograd.grad(loss, sample)[0]
    adj_grad = torch.autograd.grad(loss, modified_adj)[0]

  # Update perturbations
  lr = (num_perturbations) / (epoch + 1)
  pre_projection = int(perturbations.sum() / 2)
  perturbations = perturbations + (lr * adj_grad)
  perturbations = Utils.projection(perturbations, num_perturbations)

  # Train the model
  modified_adj = Utils.get_modified_adj(graph.adj, perturbations)
  surrogate.train1epoch(graph.features, modified_adj, graph.labels, graph.idx_train, graph.idx_test)

  t.set_postfix({"adj_l": loss.item(),
                  "adj_g": int(adj_grad.sum()),
                  "pre-p": pre_projection,
                  "target": int(num_perturbations / 2),
                  "loss": loss})

with torch.no_grad(): 
  max_loss = -1000
  for k in range(0,3):
    sample = torch.bernoulli(perturbations)
    modified_adj = Utils.get_modified_adj(graph.adj, perturbations)
    modified_adj = Utils.make_symmetric(modified_adj) # Removing this creates "impossible" adj, but works well

    predictions = surrogate(graph.features, modified_adj) 

    loss = F.cross_entropy(predictions[g0], graph.labels[g0]) \
        - F.cross_entropy(predictions[gX], graph.labels[gX])

    if loss > max_loss:
      max_loss = loss
      best = sample
      best_mod = modified_adj
  
  print(f"Best sample loss: {loss:.2f}\t Edges: {best.abs().sum() / 2:.0f}")
  
  locked_adj = Utils.get_modified_adj(graph.adj, best)

# %%
graph.adj.sum(), locked_adj.sum()

# %%
# locked_adj

################################################
# Evaluation
################################################
from Archive.Models.GCN import GCN
import Archive.Utils.Metrics as Metrics

n = 3

task_acc = {}
for task in tasks:
    m = graph.features[:,task].max().item() + 1
    m = int(m)
    task_acc[task] = {
        "dg0": 0,
        "dgX": 0
    }
    for k in range(n):
        baseline_model = GCN(
            input_features=graph.features.shape[1],
            output_classes=m,
            hidden_layers=args.hidden_layers,
            device=device,
            lr=args.model_lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            name=f"surrogate{task}"
        ).to(device)

        modified_feat = graph.features.clone()
        modified_feat[:, task] = 0
        baseline_model.fitManual(modified_feat, graph.adj, graph.features[:, task].long(), graph.idx_train, graph.idx_test, args.reg_epochs, verbose=False)
        pred = baseline_model(modified_feat, graph.adj)
        baseline_acc = Metrics.partial_acc(pred, graph.features[:, task], g0, gX, verbose=False)

        locked_model = GCN(
            input_features=graph.features.shape[1],
            output_classes=m,
            hidden_layers=args.hidden_layers,
            device=device,
            lr=args.model_lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            name=f"locked{task}"
        ).to(device)

        locked_model.fitManual(modified_feat, locked_adj, graph.features[:, task].long(), graph.idx_train, graph.idx_test, args.reg_epochs, verbose=False)
        pred = locked_model(modified_feat, locked_adj)
        locked_acc = Metrics.partial_acc(pred, graph.features[:, task], g0, gX, verbose=False)
        # # lock_gX += locked_acc["gX"]

        step_dg0 =  ((locked_acc["g0"] / baseline_acc["g0"]) - 1) / n
        step_dgX = ((locked_acc["gX"] / baseline_acc["gX"]) - 1) / n
        print(f"task:{task}\t dg0:{step_dg0:.2%} \t dgX:{step_dgX:.2%}")

        task_acc[task]["dg0"] += step_dg0
        task_acc[task]["dgX"] += step_dgX

dg0 = 0
dgX = 0
n = 3

for k in range(n):
    baseline_model = GCN(
        input_features=graph.features.shape[1],
        output_classes=graph.labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"baseline"
    ).to(device)

    baseline_model.fit(graph, args.reg_epochs, verbose=False)

    pred = baseline_model(graph.features, graph.adj)
    baseline_acc = Metrics.partial_acc(pred, graph.labels, g0, gX, verbose=False)

    locked_model = GCN(
        input_features=graph.features.shape[1],
        output_classes=graph.labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"locked"
    )

    locked_model.fitManual(graph.features, locked_adj, graph.labels, graph.idx_train, graph.idx_test, args.reg_epochs, verbose=False)

    pred = locked_model(graph.features, locked_adj)
    locked_acc = Metrics.partial_acc(pred, graph.labels, g0, gX, verbose=False)

    # base_g0 += baseline_acc["g0"]
    # base_gX += baseline_acc["gX"]
    # lock_g0 += locked_acc["g0"]
    # lock_gX += locked_acc["gX"]

    step_dg0 =  ((locked_acc["g0"] / baseline_acc["g0"]) - 1) / n
    step_dgX = ((locked_acc["gX"] / baseline_acc["gX"]) - 1) / n
    print(f"task:label\t dg0:{step_dg0:.2%} \t dgX:{step_dgX:.2%}")
    dg0 += step_dg0
    dgX += step_dgX

################################################
# Summarize
################################################

# dg0 = ((lock_g0 / base_g0) - 1) / n
# dgX = ((lock_gX / base_gX) - 1) / n

print("==== Accuracies ====")
print(f"         ΔG0\tΔGX")
print(f"task1 | {dg0:.1%}\t{dgX:.1%}")

diff = locked_adj - graph.adj
diffSummary = Metrics.show_metrics(diff, graph.labels, g0, device, verbose=False)

# print(diffSummary)

################################################
# Save
################################################

import Archive.Utils.Export as Export

def getDiff(labels, location, changeType):
    diffSummary = Metrics.show_metrics(diff, labels, g0, device, verbose=False)
    loc = diffSummary[location][changeType]
    dH = loc["same"] - loc["diff"]
    return dH

def lab_dH(location):
    return getDiff(graph.labels, location, "add") - getDiff(graph.labels, location, "remove")

def dH(task, location):
    lab = graph.features[:,task].long()
    return getDiff(lab, location, "add") - getDiff(lab, location, "remove")

results = {
    "seed": args.seed,
    "method": args.method,
    "dataset": args.dataset,
    "protect_size": args.protect_size,
    "reg_epochs": args.reg_epochs,
    "ptb_epochs": args.ptb_epochs,
    "ptb_rate": args.ptb_rate,
    "ptb_sample_num": args.num_samples,
    "ptb_sample_size": args.sample_size,
    # "ratio_g0": samplingMatrix.g0_ratio.item(),
    # "ratio_gX": samplingMatrix.gX_ratio.item(),
    # "ratio_g0gX": samplingMatrix.g0gX_ratio.item(),
    "edges": diff.abs().sum().item(),
    "base_g0": baseline_acc["g0"],
    "base_gX": baseline_acc["gX"],
    "label_d_g0": dg0,
    "label_d_gX": dgX,
    "label_dH_g0": lab_dH('g0'),
    "label_dH_gX": lab_dH('gX'),
    "label_dH_g0gX": lab_dH('g0gX'),
    "trueFeats": ' '.join(str(x.item()) for x in tasks),
}

for i, task in enumerate(task_acc):
    results[f'task{i}_d_g0'] = task_acc[task]['dg0']
    results[f'task{i}_d_gX'] = task_acc[task]['dgX']
    results[f'task{i}_dH_g0'] = dH(task, 'g0')
    results[f'task{i}_dH_gX'] = dH(task, 'gX')
    results[f'task{i}_dH_g0gX'] = dH(task, 'g0gX')


print(results)
Export.saveData('./multiTask2.csv', results)

# %%



