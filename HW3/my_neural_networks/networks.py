import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, stable_softmax, cross_entropy


class BasicNeuralNetwork:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, optimizer=None, l2_lambda=0, dropout_rate=0, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        assert dropout_rate >=0 and dropout_rate < 1, \
                "dropout rate should be in the range of [0, 1)"
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.optimizer = optimizer
        self.shape = shape
        self.gpu_id = gpu_id
        self.masks = {}
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.FloatTensor(j, i)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.FloatTensor(i, 1)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.randn(j, i).cuda(gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.randn(i, 1).cuda(gpu_id)
                           for i in self.shape[1:]]
        '''
        ### initialize velocity
        self.v_t = []
        self.v_t_b =[]
        for weight in self.weights:
            self.v_t.append(torch.zeros_like(weight))

        for bias in self.biases:
            self.v_t_b.append(torch.zeros_like(bias))
        '''
        self.iteration = 0
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.uniform_(-stdv, stdv)
            b.uniform_(-stdv, stdv)

    def _feed_forward(self, X, skip_dropout=False):
        """Forward pass
        #self.masks[i] = m

        Args:
            X: (n_neurons, n_examples)

        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        outputs = []
        act_outputs = []
        act_output = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # weighted sum
            output = torch.addmm(b, w, act_output)
            outputs.append(output)
            if i == len(self.weights) - 1:
                # in the last layer we use softmax
                act_output = stable_softmax(output)
                act_outputs.append(act_output)
            else:
                # in hidden layers we use relu
                act_output = relu(output)
                act_outputs.append(act_output)
        return outputs, act_outputs

    def _backpropagation(self, outputs, act_outputs, X, y_1hot):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """
        W_grads = []
        b_grads = []

        # get number of examples
        try: # for torch.Tensor
            m = X.shape[1]
        except: # for torch.autograd.Variable
            m = X.data.shape[1]

        # it's backpropagation, so do it in reverse order
        # assume use softmax + cross entropy loss
        
        # delta for the layer L
        delta = act_outputs[-1] - y_1hot
        for i in range(len(self.weights)-1, 0, -1):
            # calculate the gradients for W and b
            W_grad = delta.mm(act_outputs[i-1].t()) / m
            b_grad = torch.sum(delta, 1) / m

            W_grads.append(W_grad)
            b_grads.append(b_grad)

            # calculate the delta for the next layer
            delta = self.weights[i].t().mm(delta)
            # times relu derivatives
            delta[outputs[i-1] < 0] = 0

        # first hidden layer -> hidden layer
        W_grad = delta.mm(X.t()) / m
        b_grad = torch.sum(delta, 1) / m

        W_grads.append(W_grad)
        b_grads.append(b_grad)

        # reverse to forwarding order
        W_grads.reverse()
        b_grads.reverse()
        return W_grads, b_grads


    def train_one_batch(self, X, y, y_1hot):
        """Train for one batch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t_train = X.t()
        y_1hot_t_train = y_1hot.t()

        # this is for Nesterov
        self.weights, self.biases = \
                self.optimizer.ahead(self.weights, self.biases)
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # my change
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train) 
        #penalty = 
        #loss = loss + (self.l2_lambda * self.l2_penalty(self.weights))
        m = X.shape[1]
        l2_reg = None
        for W in self.weights:
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        l2_regularization = (0.5 * self.l2_lambda * l2_reg)/m
        loss = loss + l2_regularization




        # back propagation
        W_grads, b_grads = self._backpropagation(outputs,
                                                 act_outputs,
                                                 X_t_train,
                                                 y_1hot_t_train)


        self.iteration = self.iteration + 1
        # update weights and biases
        #self.weights, self.biases = \
                #self.optimizer.update(self.weights, self.biases, W_grads, b_grads)
        
        # my change
        self.weights, self.biases = \
                self.optimizer.update(self.weights, self.biases, W_grads, b_grads, self.iteration, self.l2_lambda, m )
        
        return loss

    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t, skip_dropout=True)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t)
        # my change
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        #loss = loss + (self.l2_lambda * self.l2_penalty(self.weights))

        #loss = loss + (self.l2_lambda * self.weights.norm(2))
        l2_reg = None
        for W in self.weights:
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        m = X.shape[1]
        l2_regularization = (0.5 * self.l2_lambda * l2_reg)/m
        loss = loss + l2_regularization

        # ========================================================
        # Note that you also need to change the way of calculating the loss for the L2-regularization

        return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t(), skip_dropout=True)
        return torch.max(act_outputs[-1], 0)[1]
