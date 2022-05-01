import torch
import numpy as np
from . import GCN


if __name__ == "__main__":
    a = torch.tensor([1, 1, 2, 1, 2, 1, 2])
    b = torch.tensor([1, 2, 1, 2, 2, 2, 2])

    z = chi_squared(a, b)
    print(z)



    # z = shannon_entropy(torch.tensor([1, 1, 2]))
    # print(z)