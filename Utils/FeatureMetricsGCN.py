import torch
import numpy as np
from . import GCN
from . import FeatureMetrics

def quick_learnability(features: torch.tensor, adj: torch.tensor, labels: torch.tensor, train_idx: torch.tensor, test_idx: torch.tensor) -> float:
    model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name="model"
    )
    model.fitManual(features, adj, labels, train_idx, 20, verbose=False)
    pred = model(features, adj)
    acc = FeatureMetrics.categorical_accuracy(labels[test_idx], pred[test_idx])
    return acc


if __name__ == "__main__":
    features = torch.tensor([
        [1, 0],
        [0, 1],
        [0, 1]
        ])
    adj = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0]
        ])

    z = quick_learnability(a, b, 0, torch.tensor([true, true, false]))
    print(z)



    # z = shannon_entropy(torch.tensor([1, 1, 2]))
    # print(z)