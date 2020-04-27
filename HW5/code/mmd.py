import torch


def compute_distance(X, Y):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    X : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    Y : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| X[i, :] - Y[j, :] ||_2^2``."""
    n_1, n_2 = X.size(0), Y.size(0)

    # Compute (a - b)^2 = a^2 + b^2 -2ab
    norms_1 = torch.sum(X ** 2, dim=1, keepdim=True)
    norms_2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2))
    distances_squared = norms - 2 * X @ Y.T

    # Take the absolute value due to numerical imprecision
    return torch.abs(distances_squared)

def compute_mmd(X, Y, sigma):
    """Compute MMD for the samples X and Y, using Gaussian kernel:

    k(x, x') = e^{-\frac{\lVert x - x' \rVert^2_2}{2\sigma^2}}

    Arguments
    ---------
    X: :class:`torch:torch.autograd.Variable`
        The first sample, of size ``(n, d)``.
    Y: :class:`torch:torch.autograd.Variable`
        The second sample, of size ``(m, d)``.
    sigma: :class: `float`
        The kernel parameter

    Returns
    -------
    mmd: :class:`float`
        The MMD test statistic.
    """
    mmd = 0.0

    # Combine the two samples in a single matrix and then
    # compute all the pairwise squared distances
    combined = torch.cat((X, Y), dim=0)
    distances = compute_distance(combined, combined)

    # Compute the kernel for all pairwise distances.
    # First (n x n) diagonal block has kernels for all pairs from X
    # Second (m x m) diagonal block has kernels for all pairs from Y
    # Off-diagonal (n x m) block has kernels for pairs from X and Y
    kernels = torch.exp( - distances / (2 * sigma ** 2))

    # >>>>> YOUR CODE STARTS HERE <<<<<
    batch_size = int(X.size()[0])
    #print("batch_size",batch_size)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    mmd = torch.mean(XX + YY - XY -YX)
    #print("mmd", mmd)
    # >>>>> YOUR CODE ENDS HERE <<<<<

    return mmd