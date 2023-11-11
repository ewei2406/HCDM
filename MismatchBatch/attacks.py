from deeprobust.graph.defense import GCN
import torch
from tqdm import tqdm
import torch.nn.functional as F
import utils
import numpy as np


def PGD_Attack(A, X, ε, T, train_mask, val_mask, surrogate_lr=1e-2, epochs=30, device='cpu', attack_lr=1.0):
    # A = adj
    # ε = int(0.5 * adj.sum().item())
    # X = feat
    # T = label
    # surrogate_lr = 1e-2
    # epochs = 30

    M = torch.zeros_like(A).float().to(device)
    θ = GCN(nfeat=X.shape[1], nclass=T.max().item()+1, nhid=32, lr=surrogate_lr, device=device).to(device)
    # θ.fit(X, A, T, train_mask, val_mask, train_iters=100)

    t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    A_p = torch.zeros_like(A).to(device).requires_grad_(True) # Initialize modified adj

    for epoch in t:
        pred = θ(X, A_p)
        L = F.cross_entropy(pred.squeeze(), T)
        A_grad = torch.autograd.grad(L, A_p)[0]
        lr = attack_lr / np.sqrt(epoch + 1)
        M.data.add_(lr * A_grad)
        M = utils.truncate(M, A)
        M = utils.projection(M, ε)
        A_p = utils.xor(A, utils.discretize(M)).to(device).requires_grad_(True)
        θ.fit(X, A_p, T, train_mask, val_mask, train_iters=1)

        t.set_postfix({
            "loss": L.item(),
        })

    best = torch.tensor([0])
    max_loss = -10000
    for i in range(10):
        A_p = utils.xor(A, utils.discretize(M)).to(device).requires_grad_(True)
        pred = θ(X, A_p)
        L = F.cross_entropy(pred.squeeze(), T)

        if L.item() > max_loss:
            max_loss = L.item()
            best = A_p

    best.requires_grad_(False)
    return best


def Partition_PGD(k, A, X, ε, T, train_mask, val_mask, surrogate_lr=1e-2, epochs=30, device='cpu'):
    size = int(A.shape[0] / k)

    batched_A_p = A.clone()

    for i in range(k):
        sub_A = A[size * i : size * (i + 1)][:,size * i : size * (i + 1)]
        sub_X = X[size * i : size * (i + 1)]
        sub_T = T[size * i : size * (i + 1)]
        sub_tmask = train_mask[size * i : size * (i + 1)]
        sub_vmask = val_mask[size * i : size * (i + 1)]

        sub_best = PGD_Attack(sub_A, sub_X, ε / k, sub_T, sub_tmask, sub_vmask, surrogate_lr=surrogate_lr, epochs=epochs, device=device)

        batched_A_p[size * i : size * (i + 1)][:,size * i : size * (i + 1)] = sub_best
    
    return batched_A_p

def Batch_PGD(size, n, A, X, ε, T, train_mask, val_mask, surrogate_lr=1e-2, epochs=30, device='cpu'):
    batched_A_p = A.clone()

    for i in range(n):
        indices = torch.randperm(A.shape[0])[:size]
        sub_A = A[indices][:,indices]
        sub_X = X[indices]
        sub_T = T[indices]
        sub_tmask = train_mask[indices]
        sub_vmask = val_mask[indices]

        sub_best = PGD_Attack(sub_A, sub_X, ε / n, sub_T, sub_tmask, sub_vmask, surrogate_lr=surrogate_lr, epochs=epochs, device=device)

        batched_A_p[indices][:,indices] = sub_best
    
    return batched_A_p
