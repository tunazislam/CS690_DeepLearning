"""
© 2018 Jianfei Gao
© 2020 Leonardo Teixeira

"""
import argparse
import numpy as np
import torch

from data import CoraDataset
from model import GCN2, GCN6, JanossyPool
from utils import random_seed, accuracy, set_num_samples_janossy

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

    return dataset, train_indices, valid_indices, test_indices


def prepare_model(aggregator, num_layers, num_feats, num_hidden, num_labels, lr, l2):
    r"""Prepare model

    Args
    ----
    aggregator : Str
        Name of aggregator.
    num_layers : Int
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
    GCNs = {2: GCN2, 6: GCN6}
    model = GCNs[num_layers](
        aggregator, num_feats, num_hidden, num_labels, act='relu', dropout=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer


def train(dataset, indices, model, criterion, optimizer, clip):
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
    clip : Float or None
        Gradient clipping scalor.

    """
    model.train()
    set_num_samples_janossy(model, 1)
    np.random.shuffle(indices)
    for i in range(len(indices) // 32):
        batch_indices = indices[i * 32:i * 32 + 32]
        optimizer.zero_grad()
        output = model(dataset.adj_list, dataset.feat_mx, batch_indices)
        loss = criterion(output, dataset.label_mx[batch_indices])
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            pass
        optimizer.step()


def evaluate(dataset, indices, model, criterion, num_samples=1):
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
    num_samples : Int
        How many samples (permutations) to use, for Janossy Pooling aggregator

    Returns
    -------
    loss : Float
        Averaged loss.
    acc : Float
        Averaged accuracy.

    """
    model.eval()
    set_num_samples_janossy(model, num_samples)
    output = model(dataset.adj_list, dataset.feat_mx, indices)
    loss = criterion(output, dataset.label_mx[indices])
    acc = accuracy(output, dataset.label_mx[indices])
    return float(loss.data.item()), float(acc)


def fit(dataset, train_indices, valid_indices,
        model, criterion, optimizer,
        num_epochs, clip, save, device):
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
    clip : Float or None
        Gradient clipping scalor.
    save : Str
        Path to save best model parameters.
    device : Int or None
        GPU card to work on.
    num_samples : Int
        How many samples (permutations) to use with Janossy aggregator

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
        train(dataset, train_indices, model, criterion, optimizer, clip)
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
    parser.add_argument('--agg', choices=('mean', 'LSTM'), help="Aggregator to use [default: mean]")
    parser.add_argument('--num_layers', choices=(2, 20), type=int, default=2, help="Number of graph convolutional layers [default: 2]")
    parser.add_argument('--num_hidden', type=int, default=16, help="Number of neurons of each hidden layer [default: 16]")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate [default: 0.01]")
    parser.add_argument('--l2', type=float, default=5e-4, help="L2 regularization strength [default: 5e-4]")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs [default: 100]")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples for LSTM aggregator at test time [default: 5]")
    parser.add_argument('--clip', type=float, help='Gradient clipping scale')
    parser.add_argument('--save', default="./model.pth", help="Path to save model parameters [default: ./model.pth]")
    parser.add_argument('--device', help="GPU card ID use. If not given, use CPU.")
    parser.add_argument('--seed', default=42, help="Random seed [default: 42]")
    args = parser.parse_args()

    random_seed(args.seed)

    dataset, train_indices, valid_indices, test_indices = prepare_data(args.root)

    num_feats = dataset.feat_mx.shape[1]
    num_labels = dataset.label_mx.max().item() + 1
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    model, criterion, optimizer = prepare_model(
        args.agg,
        args.num_layers,
        num_feats,
        args.num_hidden,
        num_labels,
        args.lr,
        args.l2,
    )


    fit(
        dataset,
        train_indices,
        valid_indices,
        model,
        criterion,
        optimizer,
        args.num_epochs,
        args.clip,
        args.save,
        args.device,
    )

    model.load_state_dict(torch.load(args.save))
    save_plots(train_accs, train_losses, val_accs, val_losses)
    test_loss, test_acc = evaluate(dataset, test_indices, model, criterion, args.num_samples)
    print('=' * 24)
    print("Test Loss     : {:.6f}".format(test_loss))
    print("Test Accuracy : {:.3f}%".format(test_acc))
    print('=' * 24)
