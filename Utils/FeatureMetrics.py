import torch
import numpy as np

def categorical_accuracy(truth: torch.int, guess: torch.int) -> float:
    """
    Returns the accuracy of guesses as the percent correct
    """
    correct = (guess == truth).sum().item()
    acc = correct / guess.size(dim=0)
    return acc


def discretize(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Discretizes a tensor by the number of bins
    """
    if ((not tensor_a.is_floating_point()) and (not tensor_a.is_complex())) and not force_bins:
        return tensor_a - tensor_a.min()
    
    tensor_a_max = tensor_a.max().item()
    tensor_a_min = tensor_a.min().item()
    d = tensor_a_max - tensor_a_min

    boundaries = torch.arange(start=tensor_a_min, end=tensor_a_max, step = d / n_bins)
    bucketized = torch.bucketize(tensor_a, boundaries, right=True)
    result = bucketized - bucketized.min()
    assert result.shape[0] == tensor_a.shape[0]
    return result


def dist(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns the distribution of frequencies of values in a discretized tensor
    """
    
    if tensor_a.nelement() == 0: 
        return 0
    if ((not tensor_a.is_floating_point()) and (not tensor_a.is_complex())) and not force_bins:
        offset = tensor_a.min().item()
        dist = torch.zeros([tensor_a.max().item() - offset + 1])
        for f in tensor_a:
            dist[f - offset] += 1
    else:
        if force_bins:
            tensor_a = tensor_a.float()
        dist = torch.histc(tensor_a, bins=n_bins)
    
    return dist


def p_dist(tensor_a: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns the distribution of values in a tensor as a vector of probabilities
    If the tensor is not discrete, bin using n_bins
    """
    dist = dist(tensor_a, n_bins=n_bins, force_bins=force_bins)
    return dist / dist.sum()


def joint_pdf(tensor_a: torch.tensor, tensor_b: torch.tensor, n_bins=50, force_bins=False) -> torch.tensor:
    """
    Returns a m*n tensor of joint probabilities for a and b
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]

    a_binned = discretize(tensor_a, n_bins=n_bins, force_bins=force_bins)
    b_binned = discretize(tensor_b, n_bins=n_bins, force_bins=force_bins)
    cumulative = torch.zeros([a_binned.max() + 1, b_binned.max() + 1])
    for i in range(a_binned.shape[0]):
        cumulative[a_binned[i]][b_binned[i]] += 1

    return cumulative / cumulative.sum()


def shannon_entropy(tensor_a: torch.tensor, n_bins=50) -> float:
    """
    Returns the Shannon (information) entropy of a tensor in bits (b=2)
    if tensor is an int tensor, n_bins is ignored
    """
    dist = p_dist(tensor_a)
    return ((dist * torch.log2(dist)).nan_to_num().sum() * -1).item()


def pearson_r(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the pearson r correlation coefficient of two tensors
    """
    assert tensor_a.shape[0] == tensor_b.shape[0]
    cat = torch.cat((tensor_a.unsqueeze(0), tensor_b.unsqueeze(0)))
    return torch.corrcoef(cat)[0][1].item()


def information_gain(tensor_a: torch.tensor, tensor_b: torch.tensor, discrete=True, n_bins=10) -> float:
    """
    Returns the information gain of a with respect to b
    IG = Entropy(parent) - M_Entropy(children)
    The children are split by integer value (if tensor is an int tensor) or by n_bins
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]

    parent_entropy = shannon_entropy(tensor_a)
    children_entropy = 0

    tensor_b_max = tensor_b.max().item()
    tensor_b_min = tensor_b.min().item()
    d = tensor_b_max - tensor_b_min

    if (not tensor_b.is_floating_point()) and (not tensor_b.is_complex()):
        for i in range(tensor_b_min, tensor_b_max + 1):
            split = tensor_b == i
            weight = split.sum().item() / tensor_b.shape[0]
            children_entropy += shannon_entropy(tensor_a[split]) * weight
    else:
        step_size = d / n_bins
        for i in np.arange(tensor_b_min, tensor_b_max + 1, step_size):
            split = (tensor_b - i).abs() < step_size / 2
            weight = split.sum().item() / n_bins
            children_entropy += shannon_entropy(tensor_a[split]) * weight
    
    return parent_entropy - children_entropy


def mutual_information(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the mutual information between a and b
    """

    assert tensor_a.shape[0] == tensor_b.shape[0]

    j_pdf = joint_pdf(tensor_a, tensor_b)
    cumulative = 0
    sum_X = torch.sum(j_pdf, 1)
    sum_Y = torch.sum(j_pdf, 0)
    log_pY = torch.log2(sum_Y)

    print(j_pdf)
    print(sum_X)
    print(sum_Y)
    print("")

    for idx_X in range(j_pdf.shape[0]):
        for idx_Y in range(j_pdf.shape[1]):
            p_xy = j_pdf[idx_X][idx_Y]
            cumulative += (p_xy * torch.log2(p_xy / (sum_X[idx_X] * sum_Y[idx_Y]))).nan_to_num()

    return cumulative


def chi_squared(tensor_a: torch.tensor, tensor_b: torch.tensor) -> float:
    """
    Returns the chi-sqaured statistic (WITHOUT CONTINUITY CORRECTION) of two variables
    """
    assert tensor_a.shape[0] == tensor_b.shape[0]

    j_pdf = joint_pdf(tensor_a, tensor_b)
    print(j_pdf)
    cumulative = 0
    sum_X = torch.sum(j_pdf, 0)
    sum_Y = torch.sum(j_pdf, 1)
    
    for i in range(j_pdf.shape[0]):
        E_X = sum_X * sum_Y[i]
        print((((j_pdf[i] - E_X) ** 2) / E_X))
        cumulative += (((j_pdf[i] - E_X) ** 2) / E_X).sum().item()
    
    return cumulative


if __name__ == "__main__":
    a = torch.tensor([1, 1, 2, 1, 2, 1, 2])
    b = torch.tensor([1, 2, 1, 2, 2, 2, 2])

    z = chi_squared(a, b)
    print(z)



    # z = shannon_entropy(torch.tensor([1, 1, 2]))
    # print(z)