import torch
import numpy as np
from . import Utils
from .Dataset import Dataset

class SparseGraph:
    def __init__(self, sparse_adj, labels, features, idx_train, idx_val, idx_test, split_seed):
        self.sparse_adj = sparse_adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.split_seed = split_seed

    def __repr__(self):
        return f"<Graph {self.features.shape[0]}x{self.features.shape[1]}>"

    def summarize(self, name=""):
        print()
        print(f'==== Dataset Summary: {name} ====')
        print(f'feature shape: {list(self.features.shape)}')
        print(f'num labels: {self.labels.max().item()+1}')
        print(f'split seed: {self.split_seed}')
        print(
            f'train|val|test: {self.idx_train.sum()}|{self.idx_val.sum()}|{self.idx_test.sum()}')
    
    def split(self, nsplits):
        indices = torch.zeros(10, dtype=torch.bool)
        return None

    def numEdges(self):
        return self.sparse_adj.length

    def numNodes(self):
        return self.features.shape[0]

    def getSample(self, size):
        indices = (torch.bernoulli(torch.empty(1, size)[0].uniform_(0,1))) > 0.5
        maskA = indices.nonzero().t()[0]
        maskB = (~indices).nonzero().t()[0]

        return maskA, maskB

    def getSubgraph(self, indices):
        return Graph(
            adj=self.adj[indices].t()[indices].t(),
            features=self.features[indices],
            labels=self.labels[indices],
            idx_train=self.idx_train[indices],
            idx_val=self.idx_val[indices],
            idx_test=self.idx_test[indices],
            nodeid=indices
        )

def getGraph(root, name, setting, seed, device, verbose=True):
    data = Dataset(root, name, setting, seed)
    return data

if __name__ == "__main__":
    a = getGraph("./temp", "cora", "gcn", 123, "cpu")
    print(a)