import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

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
parser.add_argument('--ptb_rate', type=float, default=0.25, help='Perturbation rate (percentage of available edges)')

parser.add_argument('--do_sampling', type=str, default='Y', help='to do sampling or not')
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

from Archive.Utils import GraphData

print(f'==== Dataset: {args.dataset} ====')

graph = GraphData.getGraph("../../Datasets", args.dataset, "gcn", args.seed, device)
graph.summarize()

################################################
# Designate protected
################################################

g0 = torch.rand(graph.features.shape[0]) <= args.protect_size
# g0 = graph.labels == 5 
g0 = g0.to(device)
gX = ~g0

print(f"Number of protected nodes: {g0.sum():.0f}")
print(f"Protected Size: {g0.sum() / graph.features.shape[0]:.2%}")

################################################
# Sampling Matrix
################################################

from Archive.Utils import Utils as Utils
from SamplingMatrix import SamplingMatrix

samplingMatrix = SamplingMatrix(g0, gX, graph.adj, args.sample_size)

samplingMatrix.get_sample()
samplingMatrix.getRatio()

################################################
# Generate Perturbations
################################################

import torch.nn.functional as F

noise = torch.zeros_like(graph.adj)
noise.index_fill_()

################################################
# Evaluation
################################################

import Archive.Utils.Metrics as Metrics

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

baseline_model.fit(graph, args.reg_epochs)

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

locked_model.fitManual(graph.features, locked_adj, graph.labels, graph.idx_train, graph.idx_test, args.reg_epochs)

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

diff = locked_adj - graph.adj
diffSummary = Metrics.show_metrics(diff, graph.labels, g0, device)

print(diffSummary)

################################################
# Check universal protection
################################################

if (args.check_universal == "Y"):
    import Archive.Utils.Export as Export
    from datetime import datetime

    universalResults = {
        "date": datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        "seed": args.seed,
        "dataset": args.dataset,
        "protect_size": args.protect_size,
        "reg_epochs": args.reg_epochs,
        "ptb_epochs": args.ptb_epochs,
        "ptb_rate": args.ptb_rate,
        "ptb_sample_num": args.num_samples,
        "ptb_sample_size": args.sample_size,
        "base_g0": baseline_acc["g0"],
        "base_gX": baseline_acc["gX"],
        "d_g0": dg0,
        "d_gX": dgX,
    }
    t = 0
    for idx in torch.randint(0, graph.features.shape[1], [5]):
        taskLabels, taskFeatures = Utils.get_task(idx, graph.features, device)

        tempModelBase = GCN(
            input_features=taskFeatures.shape[1],
            output_classes=taskLabels.max().item()+1,
            hidden_layers=args.hidden_layers,
            device=device,
            lr=args.model_lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            name=f"task {idx:.0f}"
        )

        tempModelBase.fitManual(taskFeatures, graph.adj, taskLabels, graph.idx_train, graph.idx_test, args.reg_epochs)

        pred = tempModelBase(taskFeatures, graph.adj)
        tempAccBase = Metrics.partial_acc(pred, taskLabels, g0, gX, verbose=False)

        tempModel = GCN(
            input_features=taskFeatures.shape[1],
            output_classes=taskLabels.max().item()+1,
            hidden_layers=args.hidden_layers,
            device=device,
            lr=args.model_lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            name=f"task {idx:.0f} locked"
        )

        tempModel.fitManual(taskFeatures, locked_adj, taskLabels, graph.idx_train, graph.idx_test, args.reg_epochs)

        pred = tempModel(taskFeatures, locked_adj)
        tempAcc = Metrics.partial_acc(pred, taskLabels, g0, gX, verbose=False)

        d_g0 = tempAcc["g0"] - tempAccBase["g0"]
        d_gX = tempAcc["gX"] - tempAccBase["gX"]

        print(f"d_g0: {d_g0:.2%}")
        print(f"d_gX: {d_gX:.2%}")

        universalResults[f"t{t:.0f}_task"] = idx
        universalResults[f"t{t:.0f}_g0"] = tempAccBase["g0"]
        universalResults[f"d_t{t:.0f}_g0"] = d_g0
        universalResults[f"t{t:.0f}_gX"] = tempAccBase["gX"]
        universalResults[f"d_t{t:.0f}_gX"] = d_gX

        t += 1
    
    sp = args.save_location.split(".")
    sp = sp[:-1] + ["Universal"] + sp[-1:]
    sp = ".".join(sp)
    Export.saveData(sp, universalResults)


################################################
# Save
################################################

if (args.save == "Y"):
    import Archive.Utils.Export as Export
    from datetime import datetime

    def getDiff(location, changeType):
        num = diffSummary[location][changeType]["total"]
        if num == 0: return f"{num:.0f}"
        pct = diffSummary[location][changeType]["same"] / num
        return f"{num:.0f} ({pct:.2%} similar)"

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
        "base_g0": baseline_acc["g0"],
        "base_gX": baseline_acc["gX"],
        "d_g0": dg0,
        "d_gX": dgX,
        "edges": diff.abs().sum().item(),
        "add_g0": getDiff("g0", "add"),
        "add_gX": getDiff("gX", "add"),
        "add_g0gX": getDiff("g0gX", "add"),
        "remove_g0": getDiff("g0", "remove"),
        "remove_gX": getDiff("gX", "remove"),
        "remove_g0gX": getDiff("g0gX", "remove"),

    }

    Export.saveData(args.save_location, results)