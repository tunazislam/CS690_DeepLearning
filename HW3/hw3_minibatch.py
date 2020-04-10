"""
Deep Learning @ Purdue

Author: Tunazzina Islam, Bruno Ribeiro
"""


import argparse
import logging
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_neural_networks import utils, networks, mnist, CNN_networks, shuffled_labels
from my_neural_networks.minibatcher import MiniBatcher
from my_neural_networks.optimizers import SGDOptimizer, MomentumOptimizer, NesterovOptimizer, AdamOptimizer


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    parser.add_argument('data_folder', metavar='DATA_FOLDER',
                        help='the folder that contains all the input data')

    parser.add_argument('-e', '--n_epochs', type=int, default=100,
                        help='number of epochs (DEFAULT: 100)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id to use. -1 means cpu (DEFAULT: -1)')
    parser.add_argument('-n', '--n_training_examples', type=int, default=-1,
                        help='number of training examples used. -1 means all. (DEFAULT: -1)')

    parser.add_argument('-m', '--minibatch_size', type=int, default=-1,
                        help='minibatch_size. -1 means all. (DEFAULT: -1)')
    parser.add_argument('-p', '--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adam'], default='sgd',
                        help='stochastic gradient descent optimizer (DEFAULT: sgd)')
    parser.add_argument('-r', '--l2_lambda', type=float, default=0,
                        help='the co-efficient of L2 regularization (DEFAULT: 0)')
    parser.add_argument('-o', '--dropout_rate', type=float, default=0,
                        help='dropout rate for each layer (DEFAULT: 0)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')

    ##my change for CNN
    parser.add_argument('-c', '--CNN', type=int, choices=[0, 1], default=0,
                        help='Run CNN (default: 0)')

    ##my change for shuffleCNN
    parser.add_argument('-s', '--shuffleCNN', type=int, choices=[0, 1], default=0,
                        help='Run CNN (default: 0)')

    args = parser.parse_args(argv)
    return args


def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot


def save_plots(losses, train_accs, test_accs):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, losses, '--', linewidth=2, label='loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='lower right')
    plt.savefig('loss.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')


def create_model(shape):
    if args.optimizer == 'sgd':
        optimizer = SGDOptimizer(args.learning_rate)
    elif args.optimizer == 'momentum':
        optimizer = MomentumOptimizer(args.learning_rate,
                                      shape,
                                      gpu_id=args.gpu_id,
                                      rho=0.9)
    elif args.optimizer == 'nesterov':
        optimizer = NesterovOptimizer(args.learning_rate,
                                      shape,
                                      gpu_id=args.gpu_id,
                                      rho=0.9)
    elif args.optimizer == 'adam':
        optimizer = AdamOptimizer(args.learning_rate,
                                  shape,
                                  gpu_id=args.gpu_id,
                                  beta1=0.9,
                                  beta2=0.999)
    else:
        raise NotImplementedError("Only support momentum, nesterov, or adam")

    #if (args.CNN == 1):
        #model = CNN_networks.Net()                                                                                
                                             
    #else:
    model = networks.BasicNeuralNetwork(shape,optimizer=optimizer,l2_lambda=args.l2_lambda,dropout_rate=args.dropout_rate,gpu_id=args.gpu_id)
                                                                                                                                                                        
    return model


def main():
    # DEBUG: fix seed
    # torch.manual_seed(29)

    # load data
    X_train, y_train = mnist.load_train_data(args.data_folder, max_n_examples=args.n_training_examples)
    X_test, y_test = mnist.load_test_data(args.data_folder)

    # reshape the images into one dimension
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train_1hot = one_hot(y_train, mnist.N_CLASSES)
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test_1hot = one_hot(y_test, mnist.N_CLASSES)

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)

    # get network shape
    shape = [X_train.shape[1], 300, 100, mnist.N_CLASSES]
    n_examples = X_train.shape[0]
    #logging.info("X_train shape: {}".format(X_train.shape))
    #logging.info("X_test shape: {}".format(X_test.shape))
    
    # if gpu_id is specified
    if args.gpu_id != -1:
        # move all variables to cuda
        X_train = X_train.cuda(args.gpu_id)
        y_train = y_train.cuda(args.gpu_id)
        y_train_1hot = y_train_1hot.cuda(args.gpu_id)
        X_test = X_test.cuda(args.gpu_id)
        y_test = y_test.cuda(args.gpu_id)
        y_test_1hot = y_test_1hot.cuda(args.gpu_id)


    #if args.CNN != 1:
        #print("hi")
    # create model
    model = create_model(shape)

    #if (args.CNN != 1):
    # start training
    losses = []
    train_accs = []
    test_accs = []

    # ======================================================================
    ## Model Training
    if args.minibatch_size > 0:
        
        # Please write your code here to complete the minibatch training.

        for i_epoch in range(args.n_epochs):
            logging.info("---------- EPOCH {} ----------".format(i_epoch))
            t_start = time.time()
            batcher = MiniBatcher(args.minibatch_size, n_examples ) #my change
            #print("batcher", batcher, type(batcher))
            train_idxs = batcher.get_one_batch()
            #print("train_idxs", train_idxs, type(train_idxs))

            # fit to the training data
            loss = model.train_one_batch(X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs])

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = utils.accuracy(y_train, y_train_pred)
            test_acc = utils.accuracy(y_test, y_test_pred)
            logging.info("loss = {}".format(loss))
            losses.append(loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))
            logging.info("Elapse {} seconds".format(time.time() - t_start))

        
    else:
        # here we provide the full-batch version
        for i_epoch in range(args.n_epochs):
            logging.info("---------- EPOCH {} ----------".format(i_epoch))
            t_start = time.time()

            train_idxs = np.arange(n_examples)
            np.random.shuffle(train_idxs)
            train_idxs = torch.LongTensor(train_idxs)
            print("train_idxs", type(train_idxs), len(train_idxs))
            # numpy to torch
            if args.gpu_id != -1:
                train_idxs = train_idxs.cuda(args.gpu_id)

            # fit to the training data
            loss = model.train_one_batch(X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs])

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = utils.accuracy(y_train, y_train_pred)
            test_acc = utils.accuracy(y_test, y_test_pred)
            logging.info("loss = {}".format(loss))
            losses.append(loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))
            logging.info("Elapse {} seconds".format(time.time() - t_start))

    # ======================================================================
    
    # plot
    save_plots(losses, train_accs, test_accs)

 

if __name__ == '__main__':
    args = utils.bin_config(get_arguments)
    if args.CNN != 1 and args.shuffleCNN != 1:
        main()
    
    if args.CNN == 1 and args.shuffleCNN != 1:
        CNN_networks.main()
    
    if args.shuffleCNN == 1:
        shuffled_labels.main()
   
