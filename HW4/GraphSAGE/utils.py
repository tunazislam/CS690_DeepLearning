"""© 2018 Jianfei Gao All Rights Reserved"""
import random
import numpy as np
import torch


def random_seed(seed):
    r"""Set Random Seed

    Args
    ----
    seed : Int
        Random seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        pass

def accuracy(output, target):
    r"""Compute accuracy

    Args
    ----
    output : torch.Tensor
        Label probability matrix of all nodes.
        It should be of the shape (#nodes, #labels).
    target : torch.Tensor
        Label matrix of all nodes.
        It should be of the shape (#nodes,).

    Returns
    -------
    acc : Float
        Accuracy.

    """
    output_label = output.argmax(dim=1)
    num_hit = torch.eq(output_label, target).sum().cpu().item()
    num_all = len(target)
    return num_hit / num_all * 100


def set_num_samples_janossy(model, num_samples):
    """Set the number of samples (permutations) to use in all the Janossy
    Pooling layers in the model"""
    from model import JanossyPool

    for m in model.modules():
        if isinstance(m, JanossyPool):
            m.set_num_samples(num_samples)
