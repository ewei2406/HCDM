import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tensor1 = torch.tensor([1, 2, 3]).to(device)
tensor2 = torch.tensor([3, 2, 1]).to(device)

for i in range(10):
    cat = torch.cat((tensor1.unsqueeze(0).cpu(), tensor2.unsqueeze(0).cpu())).numpy()
    a = np.corrcoef(cat)[0][1]

    print(a)