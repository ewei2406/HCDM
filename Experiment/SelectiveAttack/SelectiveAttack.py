import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


################################################
# Arguments
################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')

parser.add_argument('--model_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden_layers', type=int, default=32, help='Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')

parser.add_argument('--protect_size', type=float, default=0.1, help='Number of randomly chosen protected nodes')
parser.add_argument('--ptb_rate', type=float, default=0.5, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--sample_size', type=int, default=500, help='')
parser.add_argument('--num_samples', type=int, default=10, help='')


parser.add_argument('--reg_epochs', type=int, default=75, help='Epochs to train models')
parser.add_argument('--ptb_epochs', type=int, default=15, help='Epochs to perturb adj matrix')
parser.add_argument('--surrogate_epochs', type=int, default=0, help='Epochs to train surrogate before perturb')

parser.add_argument('--csv', type=str, default='', help='save the outputs to csv')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--ntasks', type=int, default=1, help='number of additional tasks')

args = parser.parse_args()

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

################################################
# Dataset
################################################

from Utils import GraphData

print(f'==== Dataset: {args.dataset} ====')

graph = GraphData.getGraph("../Datasets", args.dataset, "gcn", args.seed, "cpu")
graph.summarize()

################################################
# Designate protected
################################################

g0 = torch.rand(graph.features.shape[0]) <= args.protect_size
# g0 = graph.labels == 5 

gX = ~g0

print(f"Number of protected nodes: {g0.sum():.0f}")
print(f"Protected Size: {g0.sum() / graph.features.shape[0]:.2%}")

################################################
# Sampling Matrix
################################################

import Utils.Utils as Utils
from SamplingMatrix import SamplingMatrix

samplingMatrix = SamplingMatrix(g0, gX, graph.adj, args.sample_size)

samplingMatrix.get_sample()
samplingMatrix.getRatio()

################################################
# Surrogate Model
################################################

from Models.GCN import GCN

surrogate = GCN(
    input_features=graph.features.shape[1],
    output_classes=graph.labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device="cpu",
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name=f"surrogate"
)

################################################
# Generate Perturbations
################################################

import torch.nn.functional as F
from tqdm import tqdm

perturbations = torch.zeros_like(graph.adj).float()
count = torch.zeros_like(graph.adj).float()
num_perturbations = args.ptb_rate * graph.adj.sum()

t = tqdm(range(10), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
t.set_description("Perturbing")

for epoch in t:

    # Re-initialize adj_grad
    adj_grad = torch.zeros_like(graph.adj).float()

    # Get modified adj
    modified_adj = Utils.get_modified_adj(graph.adj, perturbations).float()

    for sample_epoch in range(10):
        # Get sample indices
        # sampled = torch.bernoulli(sampling_matrix)
        idx = samplingMatrix.get_sample()
        # print(idx)

        # Map sample to adj
        sample = modified_adj[idx[0], idx[1]].clone().detach().requires_grad_(True)
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
    if epoch % 2 == 0:
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

################################################
# Evaluation
################################################

import Utils.Metrics as Metrics

baseline_model = GCN(
    input_features=graph.features.shape[1],
    output_classes=graph.labels.max().item()+1,
    hidden_layers=args.hidden_layers,
    device=device,
    lr=args.model_lr,
    dropout=args.dropout,
    weight_decay=args.weight_decay,
    name=f"baseline"
)

baseline_model.fit(graph, 40)

pred = baseline_model(graph.features, graph.adj)
baseline_acc = Metrics.partial_acc(pred, graph.labels, g0, gX)


locked_adj = Utils.get_modified_adj(graph.adj, best)

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

locked_model.fitManual(graph.features, locked_adj, graph.labels, graph.idx_train, graph.idx_test, 40)

pred = locked_model(graph.features, locked_adj)
locked_acc = Metrics.partial_acc(pred, graph.labels, g0, gX)

################################################
# Summarize
################################################

dg0 = locked_acc["g0"] - baseline_acc["g0"]
dgX = locked_acc["gX"] - baseline_acc["gX"]

print("==== Accuracies ====")
print(f"         ΔG0\tΔGX")
print(f"task1 | {dg0:.1%}\t{dgX:.1%}")

