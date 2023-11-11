import torch
import dgl
import numpy as np

def get_modified_adj(adj, perturbations):
    tri = (adj + perturbations) - torch.mul(adj * perturbations, 2)
    return tri

def idx_to_bool(idx, max_len=None):
    """
    Converts an array of indices into a boolean array (where desired indices are True)
    """
    
    if not max_len:
        max_len = max(idx) + 1
    arr = torch.zeros(max_len)
    arr[idx] = 1
    return arr > 0

def eval_acc(model, graph_feat, graph_adj, graph_labels, mask) -> float:
    pred = model(graph_feat, graph_adj).cpu().argmax(dim=1)
    acc = pred[mask] == graph_labels[mask].cpu()
    return (acc.sum() / acc.shape[0]).item()

def discretize(tensor: torch.tensor, n_bins=50) -> torch.tensor:
    """
    Discretizes a tensor by the number of bins
    """
    bins = []
    tensor_np = tensor.numpy()
    for i in range(n_bins):
        bins.append(np.percentile(tensor_np, 100 * i / n_bins))

    binned = torch.zeros_like(tensor)
    for i in range(0, n_bins):
        if i == 0:
            lower = 0
        else:
            lower = bins[i - 1]
        upper = bins[i]
        binned[(tensor < upper) * (tensor > lower)] = i

    return binned.to(tensor.device)

def projection(perturbations, n_perturbations):
    """
    Get the projection of a perturbation matrix such that the sum over the distribution of perturbations is n_perturbations 
    """
    def bisection(perturbations, a, b, n_perturbations, epsilon):
        def func(perturbations, x, n_perturbations):
            return torch.clamp(perturbations-x, 0, 1).sum() - n_perturbations
        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(perturbations, miu, n_perturbations) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(perturbations, miu, n_perturbations)*func(perturbations, a, n_perturbations) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    # projected = torch.clamp(self.adj_changes, 0, 1)
    if torch.clamp(perturbations, 0, 1).sum() > n_perturbations:
        left = (perturbations - 1).min()
        right = perturbations.max()
        miu = bisection(perturbations, left, right, n_perturbations, epsilon=1e-5)
        perturbations.data.copy_(torch.clamp(
            perturbations.data - miu, min=0, max=1))
    else:
        perturbations.data.copy_(torch.clamp(
            perturbations.data, min=0, max=1))
    
    return perturbations


def make_symmetric(adj):
    """
    Makes adj. matrix symmetric about the diagonal and sets the diagonal to 0.
    Keeps the upper triangle.
    """
    upper = torch.triu(adj)

    lower = torch.rot90(torch.flip(
        torch.triu(adj, diagonal=1), [0]), 3, [0, 1])

    result = (upper + lower).fill_diagonal_(0)
    return result


def calc_homophily(adj: torch.tensor, labels: torch.tensor, mask: torch.tensor=None) -> float:
    """
    returns H (number of similar edge / number of edges)
    """
    if mask != None:
        adj = adj[mask, :][:, mask]
        labels = labels[mask]
        
    edges = adj.nonzero().t()
    match = labels[edges[0]] == labels[edges[1]]

    if match.shape[0] == 0: return float('NaN')
    return match.sum().item() / match.shape[0]

def inner_homophily(adj, labels, g0, gX) -> float:
    """
    returns H between regions (number of similar edge / number of edges)
    """
    masked = adj.detach().clone()
    masked[g0, :][:, g0] = 0
    masked[gX, :][:, gX] = 0

    edges = masked.nonzero().t()
    match = labels[edges[0]] == labels[edges[1]]

    return match.sum().item() / match.shape[0]


def save_as_dgl(graph, adj, g0, name, root='./locked/'):
  edges = adj.to_sparse().indices()
  d = dgl.graph((edges[0], edges[1]), num_nodes=graph.num_nodes())
  d.ndata['g0'] = g0
  dgl.data.utils.save_graphs(f'{root}{name}.bin', [d], {"glabel": torch.tensor([0])})

def load_dgl(name, root='./locked/') -> dgl.DGLGraph:
  d = dgl.load_graphs(f'{root}{name}.bin')[0][0]
  return d