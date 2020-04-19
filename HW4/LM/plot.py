import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

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


train_accs = []
train_losses = []
val_accs = []
val_losses = []
#for line in open('log_bptt35_graph', 'r').readlines():
for line in open(sys.argv[1], 'r').readlines()[1:]: #skip first line
    entity = line.split(',')
    #print (entity[0])
    train = entity[0].split(':')
    #print(train[1])
    train_loss = float(train[1].split('(')[0])
    #print(train_loss)
    train_acc = float(train[1].split('(')[1].split('%')[0])
    #print(train_acc)
    val = entity[1].split(':')
    valid_loss = float(val[1].split('(')[0])
    valid_acc = float(val[1].split('(')[1].split('%')[0])

    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(valid_acc)
    val_losses.append(valid_loss)
save_plots(train_accs, train_losses, val_accs, val_losses)