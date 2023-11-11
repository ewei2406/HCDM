import dgl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, lr, dropout, weight_decay):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dgl.nn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dgl.nn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))

        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, g, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

    def forward_positive(self, g, feat, edge_weight):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_weight)
        return h

    def fit(self, g, feat, labels, epochs: int, mask: torch.tensor=None, verbose=True) -> float:
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description("GCN Training")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(g, feat)
            if mask != None:
                loss = F.cross_entropy(predictions[mask], labels[mask])
            else:
                loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})
        
        return loss.item()


class DenseGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, lr, dropout, weight_decay):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dgl.nn.DenseGraphConv(in_size, hid_size))
        self.layers.append(dgl.nn.DenseGraphConv(hid_size, out_size))

        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, adj, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(adj, h)
        return h

    def forward_positive(self, g, feat, edge_weight):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_weight)
        return h

    def fit(self, g, feat, labels, epochs: int, mask: torch.tensor=None, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description("GCN Training")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(g, feat)
            if mask != None:
                loss = F.cross_entropy(predictions[mask], labels[mask])
            else:
                loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})

        return loss.item()