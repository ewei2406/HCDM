import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers = 2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2, allow_zero_in_degree=True))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2, allow_zero_in_degree=True))

        self.convs.append(GraphConv(out_dim * 2, out_dim, allow_zero_in_degree=True))
        self.act_fn = act_fn

    def forward(self, graph, feat, weight=None, edge_weight=None):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat, weight, edge_weight))

        return feat

# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)


class Grace(nn.Module):
    r"""
        GRACE model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    temp: float
        Temperature constant.
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))        # intra-view pairs
        between_sim = f(self.sim(z1, z2))     # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def batched_semi_loss(self, z1: th.Tensor, z2: th.Tensor,
                          batch_size: int, device=None):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        # device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: th.exp(x / self.temp)
        indices = th.arange(0, num_nodes) # .to(device)
        losses = []

        import time

        for i in range(num_batches):
            t0 = time.time()
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1)).to(device)  # [B, N]
            print(f'a1: {(time.time() - t0) * 1000}')


            between_sim = f(self.sim(z1[mask], z2)).to(device)   # [B, N]
            print(f'a2: {(time.time() - t0) * 1000}')

            losses.append(-th.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

            print(f'a3: {(time.time() - t0) * 1000}')

        return th.cat(losses)

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2, batch_size=0, device=None, g1_weights=None, g2_weights=None):
        # encoding
        if g1_weights != None and g2_weights != None:
            h1 = self.encoder(graph1, feat1, edge_weight=g1_weights)
            h2 = self.encoder(graph2, feat2, edge_weight=g2_weights)
        else:
            h1 = self.encoder(graph1, feat1)
            h2 = self.encoder(graph2, feat2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        if batch_size == 0:
            l1 = self.get_loss(z1, z2)
            l2 = self.get_loss(z2, z1)
        else:
            l1 = self.batched_semi_loss(z1, z2, batch_size, device)
            l2 = self.batched_semi_loss(z2, z1, batch_size, device)

        ret = (l1 + l2) * 0.5

        return ret.mean()