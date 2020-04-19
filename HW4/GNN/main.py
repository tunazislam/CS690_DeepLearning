"""
© 2018 Jianfei Gao
© 2020 Leonardo Teixeira
"""

import argparse


import numpy as np
import torch

from data import CoraDataset
from model import VariableGCN
from utils import random_seed, accuracy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prepare_data(root, shuffle=False):
    r"""Prepare data for different usages

    Args
    ----
    root : Str
        Root folder of cora dataset.
    shuffle : Bool
        Randomly shuffle raw data.

    Returns
    -------
    dataset : CoraDataset
        Cora dataset.
    train_indices : np.array
        An array of points ID to train on.
    valid_indices : np.array
        An array of points ID to validate on.
    test_indices : np.array
        An array of points ID to test on.

    """
    # load dataset
    dataset = CoraDataset(root)

    # separate dataset indices
    num_nodes = len(dataset.pt_str2int)
    if shuffle:
        indices = np.random.permutation(num_nodes)
    else:
        indices = np.arange(num_nodes)
    train_indices = indices[1500:]
    valid_indices = indices[1000:1500]
    test_indices = indices[0:1000]

    for i in (train_indices, valid_indices, test_indices):
        degrees = (dataset.adj_mx[i] > 0).sum(dim=1)
        print("Min: {:>2d}, Max: {:>2d}, Avg: {:>.2f}".format(degrees.min(), degrees.max(), degrees.float().mean()))

    return dataset, train_indices, valid_indices, test_indices


def prepare_model(num_layers, num_feats, num_hidden, num_labels, lr, l2):
    r"""Prepare model

    Args
    ----
    num_feats : Int
        Number of graph convolutional layers.
    num_feats : Int
        Number of input neurons.
    num_hidden : Int
        Number of hidden neurons.
    num_labels : Int
        Number of output neurons.
    lr : Float
        Learning rate.
    l2 : Float
        L2 regularization strength.

    Returns
    -------
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.

    """
    model = VariableGCN(num_feats, num_hidden, num_labels, num_layers, dropout=0)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer


def train(dataset, indices, model, criterion, optimizer):
    r"""Train an epoch

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to train on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.

    """
    model.train()
    optimizer.zero_grad()
    output = model(dataset.adj_mx, dataset.feat_mx)
    loss = criterion(output[indices], dataset.label_mx[indices])
    #acc = accuracy(output[indices], dataset.label_mx[indices])
    #print (loss, acc)
    loss.backward()
    optimizer.step()


def evaluate(dataset, indices, model, criterion):
    r"""Evaluate

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to evaluate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.

    Returns
    -------
    loss : Float
        Averaged loss.
    acc : Float
        Averaged accuracy.

    """
    model.eval()
    output = model(dataset.adj_mx, dataset.feat_mx)
    loss = criterion(output[indices], dataset.label_mx[indices])
    acc = accuracy(output[indices], dataset.label_mx[indices])
    return float(loss.data.item()), float(acc)


def fit(dataset, train_indices, valid_indices,
        model, criterion, optimizer,
        num_epochs, save, device=None):
    r"""Fit model parameters

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    train_indices : np.array
        An array of points ID to train on.
    valid_indices : np.array
        An array of points ID to validate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.
    num_epochs : Int
        Number of training epochs.
    save : Str
        Path to save best model parameters.
    device : Int or None
        GPU card to work on.

    """
    print('=' * 58)
    print("Train {} Epochs".format(num_epochs))
    print('-' * 58)

    # transfer to GPU
    if device is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
        model = model.cuda(device)
        criterion = criterion.cuda(device)
        dataset.feat_mx = dataset.feat_mx.cuda(device)
        dataset.label_mx = dataset.label_mx.cuda(device)
        dataset.adj_mx = dataset.adj_mx.cuda(device)
    else:
        pass

    # first evaluation
    train_loss, train_acc = evaluate(dataset, train_indices, model, criterion)
    valid_loss, valid_acc = evaluate(dataset, valid_indices, model, criterion)

    best_loss = valid_loss
    torch.save(model.state_dict(), save)

    print("[{:>3d}] Train: {:.6f} ({:.3f}%), Valid: {:.6f} ({:.3f}%)".format(
            0, train_loss, train_acc, valid_loss, valid_acc))

  
    # train and evaluate
    for i in range(1, num_epochs + 1):
        train(dataset, train_indices, model, criterion, optimizer)
        train_loss, train_acc = evaluate(dataset, train_indices, model, criterion)
        valid_loss, valid_acc = evaluate(dataset, valid_indices, model, criterion)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save)
        else:
            pass

        print("[{:>3d}] Train: {:.6f} ({:.3f}%), Valid: {:.6f} ({:.3f}%)".format(
                i, train_loss, train_acc, valid_loss, valid_acc))

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(valid_acc)
        val_losses.append(valid_loss)

    print('=' * 58)
    print()


def check_mixing(dataset, model):
    r"""Evaluate

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    model : torch.nn.Module
        Neural network model.

    """
    # Extract the last embedding, for all nodes in the graph
    model.eval()
    _ = model(dataset.adj_mx, dataset.feat_mx)
    embeds = torch.nn.functional.normalize(model.embeds, p=2, dim=1)

    # Compute the MSEs for every pair of nodes
    mses = []
    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            # >>>>> YOUR CODE STARTS HERE <<<<<
            #mse = <...>
            mse = ((embeds[i] - embeds[j]) ** 2).sum()
            # >>>>> YOUR CODE ENDS HERE <<<<<
            mses.append(mse)

    print("Max MSE:", max(mses))

def save_plots(train_accs, train_losses, val_accs, val_losses):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(train_losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, train_losses, '--', linewidth=2, label='train')
    ax.plot(xs, val_losses, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.savefig('loss.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, val_accs, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="./Data/cora", help="Root folder for Cora dataset [default: ./Data/cora]")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of graph convolutional layers [default: 2]")
    parser.add_argument('--num_hidden', type=int, default=16, help="Number of neurons of each hidden layer [default: 16]")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate [default: 0.01]")
    parser.add_argument('--l2', type=float, default=5e-4, help="L2 regularization strength [default: 5e-4]")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs [default: 100]")
    parser.add_argument('--save', default="./model.pth", help="Path to save model parameters [default: ./model.pth]")
    parser.add_argument('--device', help="GPU card ID use. If not given, use CPU.")
    parser.add_argument('--seed', default=42, help="Random seed [default: 42]")
    args = parser.parse_args()

    random_seed(args.seed)

    dataset, train_indices, valid_indices, test_indices = prepare_data(args.root)
    # print(dataset.feat_mx)
    # print(dataset.label_mx)
    # print(dataset.adj_mx)
    num_feats = dataset.feat_mx.shape[1]
    num_labels = dataset.label_mx.max().item() + 1
    #print("num_feats,num_labels",num_feats,num_labels) #num_feats,num_labels 1433, 7
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    model, criterion, optimizer = prepare_model(
        args.num_layers,
        num_feats,
        args.num_hidden,
        num_labels,
        args.lr,
        args.l2
    )

    fit(
        dataset,
        train_indices,
        valid_indices,
        model,
        criterion,
        optimizer,
        args.num_epochs,
        args.save,
        args.device
    )

    model.load_state_dict(torch.load(args.save))
    save_plots(train_accs, train_losses, val_accs, val_losses)
    test_loss, test_acc = evaluate(dataset, test_indices, model, criterion)
    print('=' * 24)
    print("Test Loss     : {:.6f}".format(test_loss))
    print("Test Accuracy : {:.3f}%".format(test_acc))
    print('=' * 24)

    # Compute the MSE between pairs of embeddings
    check_mixing(dataset, model)


