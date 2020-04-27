import argparse
import time

import torch

import numpy as np

from model import CNN
from mmd import compute_mmd
from data import prepare_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_one_batch(optimizer, model, X_source, y_source, X_target, reg_str, kernel_sigma, device):
    """Train for one batch, using both """
    optimizer.zero_grad()
    model.train()

    X_source = X_source.to(device)
    y_source = y_source.to(device)

    h_source, z_source = model(X_source, return_representation=True)
    loss = torch.nn.functional.cross_entropy(h_source, y_source, reduction='mean')

    if reg_str:
        X_target = X_target.to(device)
        z_target = model.representation(X_target)
        loss += reg_str * compute_mmd(z_source, z_target, kernel_sigma)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

    optimizer.step()

    return float(loss)


def evaluate(loader, model, device):
    """Compute accuracy for the entire dataset provided by the loader"""
    model.eval()
    count, correct = 0, 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            h = model(X)
            y_hat = h.argmax(dim=1)
            count += len(X)
            correct += float( (y_hat == y).float().sum() )
    return correct / count * 100.0

#def save_plots(train_accs, train_losses, val_accs, val_losses):
def save_plots(losses, accuracies, target_val_accuracies):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, losses, '--', linewidth=2, label='train')
    #ax.plot(xs, val_losses, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.savefig('loss.png')

    # plot validation accuracy for each epoch, for both source and target domain.
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, accuracies, '--', linewidth=2, label='source domain')
    ax.plot(xs, target_val_accuracies, '-', linewidth=2, label='target domain')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="./data", help="Folder containing the datasets")
    parser.add_argument("-b", "--batch_size", default=256, type=int)
    parser.add_argument("-l", "--learning_rate", default=1, type=float)
    parser.add_argument("-n", "--num_epochs", default=200, type=int)
    parser.add_argument("--reg_str", default=None, type=float)
    parser.add_argument("--kernel_sigma", default=20, type=float)
    parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use")
    args = parser.parse_args()

    if args.gpu is None or args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Get the data
    loaders = prepare_data(args.dataset, args.batch_size)

    # Initial (random) model
    random_model = CNN()

    # Prepare the model and the optimizer
    model = CNN()
    model.load_state_dict(random_model.state_dict())
    model.to(device)

    # Set up SGD with decaying learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # We'll use these variables for early stopping
    best_accuracy = 0
    best_epoch = None
    best_model = model.state_dict()

    # Training loop
    losses = []
    accuracies = []
    target_val_accuracies = [] #my change
    for epoch in range(args.num_epochs):
        timer_start = time.time()

        # Train the model for one epoch
        avg_batch_loss = None
        for batch_no, ((X_source, y_source), (X_target, _)) in enumerate(zip(loaders['source']['train'], loaders['target']['train'])):
            batch_loss = train_one_batch(optimizer, model, X_source, y_source, X_target, args.reg_str, args.kernel_sigma, device)

            # Exponential average of the batch loss
            if avg_batch_loss is None:
                avg_batch_loss = batch_loss
            else:
                avg_batch_loss = 0.2 * avg_batch_loss + 0.8 * batch_loss

        train_time = time.time() - timer_start

        # Compute validation accuracy
        timer_start = time.time()
        source_validation_accuracy = evaluate(loaders['source']['validation'], model, device)
        target_validation_accuracy = evaluate(loaders['target']['validation'], model, device)
        found_better = False
        if target_validation_accuracy > best_accuracy:
            best_accuracy = target_validation_accuracy
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, "model.pth")
            found_better = True
        eval_time = time.time() - timer_start

        # Update learning rate according to scheduler
        lr_schedule.step()

        losses.append(avg_batch_loss)
        accuracies.append(source_validation_accuracy)
        target_val_accuracies.append(target_validation_accuracy)
        better = "*" if found_better else " "
        print(f"[Epoch {epoch:3d}] Avg. batch loss: {avg_batch_loss:6.3e} :: Val. Acc (Src -> Tgt) {source_validation_accuracy:4.1f} % -> {target_validation_accuracy:4.1f} % [Time: {train_time + eval_time:4.1f} ({train_time:4.1f}) s] [{better}]")

    #plot learning curves
    save_plots(losses, accuracies, target_val_accuracies)
    model.load_state_dict(best_model)
    model = model.to(device)
    print(f"Best model found at epoch {best_epoch}, with target validation accuracy = {best_accuracy:.1f} %")

    timer_start = time.time()
    print("Computing source domain test accuracy...")
    source_accuracy = evaluate(loaders['source']['test'], model, device)
    print("Computing target domain test accuracy...")
    target_accuracy = evaluate(loaders['target']['test'], model, device)
    print("Final Test Accuracy:")
    print(f"  -- Source Domain: {source_accuracy:5.1f} %")
    print(f"  -- Target Domain: {target_accuracy:5.1f} %")
    eval_time = time.time() - timer_start
    print(f"Time spent with final evaluation: {eval_time:.1f} s")
