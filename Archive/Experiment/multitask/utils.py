import torch
import dgl

def get_modified_adj(adj, perturbations):
    tri = (adj + perturbations) - torch.mul(adj * perturbations, 2)
    return tri


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