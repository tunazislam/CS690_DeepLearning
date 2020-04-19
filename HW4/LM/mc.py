"""
© 2018 Jianfei Gao
© 2020 Leonardo Teixeira
"""
import os
import math
import argparse

from data import PTBDataset, MarkovChainLoader


def prepare_data(root, order):
    r"""Prepare data for different usages

    Args
    ----
    root : Str
        Root folder of PTB dataset.
    order : Int
        Order of Markov model.
    batch_size : Int
        Batch size.

    Returns
    -------
    train_loader : MCBatchLoader
        Data loader to train on.
    valid_loader : MCBatchLoader
        Data loader to validate on.
    test_loader : MCBatchLoader
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
    train_loader = MarkovChainLoader(train_dataset, order=order)
    valid_loader = MarkovChainLoader(valid_dataset, order=order)
    test_loader = MarkovChainLoader(test_dataset, order=order)

    return train_loader, valid_loader, test_loader, len(dictionary)


def evaluate(train_loader, loader, num_labels):
    r"""Evaluate

    Args
    ----
    train_loader : MarkovChainLoader
        Markov chain of training data.
    loader : MarkovChainLoader
        Markov chain data to evaluate on.
    num_labels : Int
        Number of different words in dictionary.

    Returns
    -------
    loss : Float
        Averaged loss.

    """
    counts = train_loader.counts
    sums = train_loader.sums
    loss_sum, loss_cnt = 0, 0
    for input, target in zip(loader.input_seq, loader.target_seq):
        input = tuple(input)
        if input in counts:
            if target in counts[input]:
                prob = counts[input][target] / (sums[input] + 1)
            else:
                prob = 1 / (sums[input] + 1) / (num_labels - len(counts[input]))
        else:
            prob = 1 / num_labels
        loss_sum += (-math.log(prob))
        loss_cnt += 1
    #print("loss_sum",loss_sum)
    #print("loss_cnt",loss_cnt)
    return loss_sum / loss_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="./Data/ptb",  help="Root folder of PTB dataset [default: ./Data/ptb]")
    parser.add_argument('--order', default=5, type=int, help="Order of Markov model [default: 5]")
    args = parser.parse_args()

    train_loader, valid_loader, test_loader, num_labels = prepare_data(args.root, args.order)

    train_loss = evaluate(train_loader, train_loader, num_labels) 
    valid_loss = evaluate(train_loader, valid_loader, num_labels) 
    test_loss = evaluate(train_loader, test_loader, num_labels)
    print('=' * 22)
    print("Train Loss: {:10.3f}".format(math.exp(train_loss)))
    print("Valid Loss: {:10.3f}".format(math.exp(valid_loss)))
    print("Test Loss : {:10.3f}".format(math.exp(test_loss)))
    print('=' * 22)
