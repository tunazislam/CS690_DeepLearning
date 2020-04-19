"""
© 2018 Jianfei Gao
© 2020 Leonardo Teixeira

Usage:
    main.py [--root <folder>] [--bptt <N>] [--batch-size <N>] [--embed <N>]
            [--hidden <N>] [--lr <lr>] [--epoch <N>] [--clip <C>] [--save <path>]
            [--device <ID>] [--seed <S>]

Options:
    --root <folder>     Root folder of PTB dataset [default: ./data/ptb]
    --bptt <N>          Truncated backpropagation trought time length [default: 35]
    --batch-size <N>    Batch size [default: 20]
    --embed <N>         Number of neurons of word embedding [default: 200]
    --hidden <N>        Number of neurons of each hidden layer [default: 200]
    --lr <lr>           Learning rate [default: 20]
    --epoch <N>         Number of training epochs [default: 10]
    --clip <C>          Gradient clipping scalor [default: 0.25]
    --save <path>       Path to save model parameters [default: ./lstm.pt]
    --device <ID>       GPU card ID (not being specified as CPU) to work on
    --seed <S>          Random seed [default: 42]

"""
import os
import math
import torch
import argparse

from data import PTBDataset, BPTTBatchLoader
from model import LSTMLM
#from utils import random_seed
from utils import random_seed, accuracy #mychange

def prepare_data(root, bptt, batch_size):
    r"""Prepare data for different usages

    Args
    ----
    root : Str
        Root folder of PTB dataset.
    bptt : Int
        Truncated backpropagation through time length.
    batch_size : Int
        Batch size.

    Returns
    -------
    train_loader : BPTTBatchLoader
        Data loader to train on.
    valid_loader : BPTTBatchLoader
        Data loader to validate on.
    test_loader : BPTTBatchLoader
        Data loader to test on.
    num_labels : Int
        Number of different words in dictionary.

    """
    # load dataset
    dictionary = {}
    train_dataset = PTBDataset(os.path.join(root, 'ptb.train.txt'), dictionary)
    valid_dataset = PTBDataset(os.path.join(root, 'ptb.valid.txt'), dictionary)
    test_dataset = PTBDataset(os.path.join(root, 'ptb.test.txt'), dictionary)

    # prepare data loader
    train_loader = BPTTBatchLoader(train_dataset, bptt=bptt, batch_size=batch_size)
    valid_loader = BPTTBatchLoader(valid_dataset, bptt=bptt, batch_size=batch_size)
    test_loader = BPTTBatchLoader(test_dataset, bptt=bptt, batch_size=1)

    return train_loader, valid_loader, test_loader, len(dictionary)

def prepare_model(num_labels, num_embeds, num_hidden, lr):
    r"""Prepare model

    Args
    ----
    num_labels : Int
        Number of different word labels.
    num_embeds : Int
        Number of embedding neurons.
    num_hidden : Int
        Number of hidden neurons.
    lr : Float
        Learning rate.

    Returns
    -------
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.

    """
    model = LSTMLM(num_labels, num_embeds, num_hidden, 2, dropout=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, criterion, optimizer

def train(loader, model, criterion, optimizer, clip, device=None):
    r"""Train an epoch

    Args
    ----
    loader : BPTTBatchLoader
        Data loader to train on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.
    clip : Float
        Gradient clipping scalor.
    device : Int or None
        GPU card to work on.

    """
    model.train()
    iterator = iter(loader)
    model.states = None
    for i in range(len(iterator)):
        input, target = next(iterator)
        if device is not None and torch.cuda.is_available():
            input = input.cuda(device)
            target = target.cuda(device)
        else:
            pass
        optimizer.zero_grad()
        output = model(input)
        output_flat = output.view(-1, output.size(2))
        target_flat = target.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

def evaluate(loader, model, criterion, device=None):
    r"""Evaluate

    Args
    ----
    loader : BPTTBatchLoader
        Data loader to evaluate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    device : Int or None
        GPU card to work on.

    Returns
    -------
    loss : Float
        Averaged loss.

    """
    model.eval()
    iterator = iter(loader)
    model.states = None
    loss_sum, loss_cnt = 0, 0
    for i in range(len(iterator)):
        input, target = next(iterator)
        if device is not None and torch.cuda.is_available():
            input = input.cuda(device)
            target = target.cuda(device)
        else:
            pass
        optimizer.zero_grad()
        output = model(input)
        output_flat = output.reshape(-1, output.size(2))
        target_flat = target.reshape(-1)
        loss = criterion(output_flat, target_flat)
        acc = accuracy(output_flat, target_flat)
        loss_sum += (loss.data.item() * input.size(0))
        loss_cnt += input.size(0)
    return loss_sum / loss_cnt, float(acc)

def fit(train_loader, valid_loader, model, criterion, optimizer,
        num_epochs, clip, save, device=None):
    r"""Fit model parameters

    Args
    ----
    train_loader : BPTTBatchLoader
        Data loader to train on.
    valid_loader : BPTTBatchLoader
        Data loader to validate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.
    num_epochs : Int
        Number of training epochs.
    clip : Float
        Gradient clipping scalor.
    save : Str
        Path to save best model parameters.
    device : Int or None
        GPU card to work on.

    """
    print('=' * 43)
    print("Train {} Epochs".format(num_epochs))
    print('-' * 43)

    # transfer to GPU
    if device is not None and torch.cuda.is_available():
        model = model.cuda(device)
        criterion = criterion.cuda(device)
    else:
        pass

    # first evaluation
    train_loss, train_acc = evaluate(train_loader, model, criterion, device=device)
    valid_loss, valid_acc = evaluate(valid_loader, model, criterion, device=device)

    best_loss = valid_loss
    torch.save(model.state_dict(), save)

    print("[{:>3d}] Train: {:10.3f}({:.3f}%) , Valid: {:10.3f}({:.3f}%)".format(
            0, math.exp(train_loss), train_acc, math.exp(valid_loss), valid_acc))

    # train and evaluate
    for i in range(1, num_epochs + 1):
        train(train_loader, model, criterion, optimizer, clip, device=device)
        train_loss, train_acc = evaluate(train_loader, model, criterion, device=device)
        valid_loss, valid_acc = evaluate(valid_loader, model, criterion, device=device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save)
        else:
            pass

        #print("[{:>3d}] Train: {:10.3f} , Valid: {:10.3f}".format(
                #i, math.exp(train_loss), math.exp(valid_loss)))
        print("[{:>3d}] Train: {:10.3f}({:.3f}%) , Valid: {:10.3f}({:.3f}%)".format(
            i, math.exp(train_loss), train_acc, math.exp(valid_loss), valid_acc))


    print('=' * 43)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./Data/ptb', help="Root folder of PTB dataset [default: ./Data/ptb]")
    parser.add_argument("--bptt", default=35, type=int, help="Truncated backpropagation trought time length [default: 35]")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size [default: 20]")
    parser.add_argument("--embed", default=200, type=int, help="Number of neurons of word embedding [default: 200]")
    parser.add_argument("--hidden", default=200, type=int, help="Number of neurons of each hidden layer [default: 200]")
    parser.add_argument("--lr", default=20, type=float, help="Learning rate [default: 20]")
    parser.add_argument("--epoch", default=10, type=int, help="Number of training epochs [default: 10]")
    parser.add_argument("--clip", default=0.25, type=float, help="Gradient clipping scalor [default: 0.25]")
    parser.add_argument("--save", default="./lstm.pth", help="Path to save model parameters [default: ./lstm.pth]")
    parser.add_argument("--device", type=int, help="GPU card ID to use (if not given, use CPU)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed [default: 42]")
    args = parser.parse_args()

    if args.device is not None and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.device}")

    random_seed(args.seed)

    train_loader, valid_loader, test_loader, num_labels = prepare_data(
        args.root, args.bptt, args.batch_size)

    num_embeds = args.embed
    num_hidden = args.hidden
    model, criterion, optimizer = prepare_model(
        num_labels, num_embeds, num_hidden, args.lr)

    num_epochs = args.epoch
    fit(train_loader, valid_loader, model, criterion, optimizer,
        num_epochs, args.clip, args.save, args.device)

    model.load_state_dict(torch.load(args.save))
    test_loss, test_acc = evaluate(test_loader, model, criterion, device=args.device)
    print('=' * 21)
    print("Test Loss: {:10.3f}".format(math.exp(test_loss)))
    print("Test Accuracy : {:10.3f}%".format(test_acc))
    print('=' * 21)
