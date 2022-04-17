# %%
import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), '../..'))

################################################
# Arguments
################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--protect_size', type=float, default=0.1, help='Number of randomly chosen protected nodes')
parser.add_argument('--ptb_rate', type=float, default=0.25, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--sample_size', type=int, default=500, help='')
parser.add_argument('--num_samples', type=int, default=20, help='')
parser.add_argument('--num_subtasks', type=int, default=10, help='')

parser.add_argument('--reg_epochs', type=int, default=100, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=30, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=0, help='Epochs to train surrogate before perturb')

parser.add_argument('--save', type=str, default='N', help='save the outputs to csv')
parser.add_argument('--save_location', type=str, default="./UniversalProtection.csv", help='where to save the outputs to csv')

args = parser.parse_args() # Remove string if file

# %%
################################################
# Environment
################################################

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

# %%
################################################
# Dataset
################################################

from Utils import GraphData
from Utils import Metrics
import numpy as np

print(f'==== Dataset: {args.dataset} ====')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)

graph.summarize()

# %%
from Utils import Utils

tasks = {}
np.seterr(invalid='ignore')

# Add labels as a task
entropy, correlation, idx = Metrics.get_ent_cor(graph.labels.unsqueeze(1).float(), graph.labels, 1)
tasks[-1] = {
        "ent": entropy.item(),
        "corr": correlation.item(),
        "feat": graph.labels
    }

# Find highest entropy
entropy, correlation, idx = Metrics.get_ent_cor(graph.features, graph.labels, args.num_subtasks)

for f_idx in range(idx.shape[0]):
    tasks[idx[f_idx].item()] = {
        "ent": entropy[f_idx].item(),
        "corr": correlation[f_idx].item(),
        "feat": graph.features[:, idx[f_idx]]
    }

# Remove from feature set
unselected = Utils.bool_to_idx(~Utils.idx_to_bool(idx, graph.features.shape[1])).squeeze()
graph.features = graph.features[:, unselected]


# %%
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

import Utils.Utils as Utils
from SamplingMatrix import SamplingMatrix

samplingMatrix = SamplingMatrix(g0, gX, graph.adj, args.sample_size)

samplingMatrix.get_sample()
samplingMatrix.getRatio()

# %%
################################################
# Surrogate Model
################################################

from Models.GCN import GCN

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

# %%
################################################
# Generate Perturbations
################################################

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

    # Do sampling
    for sample_epoch in range(args.num_samples):
        # Get sample indices
        # sampled = torch.bernoulli(sampling_matrix)
        idx = samplingMatrix.get_sample()
        # print(idx)

        # Map sample to adj
        sample = modified_adj[idx[0], idx[1]].clone().detach().requires_grad_(True).to(device)
        modified_adj[idx[0], idx[1]] = sample

        # Get grad
        predictions = surrogate(graph.features, modified_adj)
        loss = F.cross_entropy(predictions[g0], graph.labels[g0]) \
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



# %%
################################################
# Get best sample
################################################

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
    
    print(f"Best sample loss: {loss:.2f}\t Edges: {best.abs().sum() / 2:.0f}")



# %%
################################################
# Evaluate universal
################################################

from Utils import Export

print("==== Eval ====")
print(f"Task,\tΔG0,\tΔGX,\tEnt\tCorr")

locked_adj = Utils.get_modified_adj(graph.adj, best)
diff = locked_adj - graph.adj

for t in tasks:
    temp_labels = tasks[t]["feat"].long()
    label_max = temp_labels.max().item() + 1

    baseline_model = GCN(
        input_features=graph.features.shape[1],
        output_classes=label_max,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"baseline"
    ).to(device)

    baseline_model.fitManual(graph.features, graph.adj, temp_labels, graph.idx_train, graph.idx_test, args.reg_epochs, False)

    pred = baseline_model(graph.features, graph.adj)
    baseline_acc = Metrics.partial_acc(pred, temp_labels, g0, gX, False)


    locked_model = GCN(
        input_features=graph.features.shape[1],
        output_classes=label_max,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name=f"locked"
    )

    locked_model.fitManual(graph.features, locked_adj, temp_labels, graph.idx_train, graph.idx_test, args.reg_epochs, False)

    pred = locked_model(graph.features, locked_adj)
    locked_acc = Metrics.partial_acc(pred, temp_labels, g0, gX, False)

    dg0 = locked_acc["g0"] - baseline_acc["g0"]
    dgX = locked_acc["gX"] - baseline_acc["gX"]

    print(f"{t},\t{dg0:.1%},\t{dgX:.1%},\t{tasks[t]['ent']:.2f},\t{tasks[t]['corr']:.2f}")

    diffSummary = Metrics.show_metrics(diff, temp_labels, g0, device, False)

    results = {
        "seed": args.seed,
        "dataset": args.dataset,
        "protect_size": args.protect_size,
        "reg_epochs": args.reg_epochs,
        "ptb_epochs": args.ptb_epochs,
        "ptb_rate": args.ptb_rate,
        "ptb_sample_num": args.num_samples,
        "ptb_sample_size": args.sample_size,
        "ratio_g0": samplingMatrix.g0_ratio.item(),
        "ratio_gX": samplingMatrix.gX_ratio.item(),
        "ratio_g0gX": samplingMatrix.g0gX_ratio.item(),
        "feature": t,
        "corr": tasks[t]["corr"],
        "entropy": tasks[t]["ent"],
        "base_g0": baseline_acc["g0"],
        "base_gX": baseline_acc["gX"],
        "d_g0": dg0,
        "d_gX": dgX,
        "edges": int(diff.abs().sum().item()),
    }

    for add_remove in ["add", "remove"]:
        for location in ["g0", "gX", "g0gX"]:
            for similar in ["same", "diff"]:
                results["_".join([add_remove, location, similar])] = diffSummary[location][add_remove][similar]

    Export.saveData(args.save_location, results)
