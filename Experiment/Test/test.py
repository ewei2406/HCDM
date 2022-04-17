import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import torch

vals = torch.tensor([1, 2, 3, 2, 1, 2, 3, 3])

torch.distributions.distribution.Distribution()