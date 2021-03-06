
import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, softmax, cross_entropy, sigmoid, stable_softmax
from torch.autograd import Variable


class AutogradNeuralNetwork:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        #print("shape", shape, len(shape)) #shape [784, 300, 100, 10] 4
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.autograd.Variable(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.autograd.Variable(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        self.init_weights()


    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        #print (len(self.weights)) # len(self.weights) = 3
        #print ("weight init", self.weights)
        for i in range(len(self.weights)):
            #print (self.weights[i], len(self.weights[i]))
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        w1x = torch.matmul(self.weights[0], X)  #input layer (wx)
        #print ("before bias wx1", self.wx1, self.wx1.size())
        z1 = torch.add(w1x, self.biases[0]) # input times first layer matrices   (wx + bias)
        #print ("z1",z1, z1.size())      
        h1 = relu(z1) # ReLU activation function 1st hidden layer. output of 1st hidden layer
        #print ("h1",h1, h1.size())
        
       
        w2h1 = torch.matmul(self.weights[1], h1 ) #hidden layer2 (wx)
        z2 = torch.add(w2h1, self.biases[1]) #hidden layer2  (wx + bias)
        h2 = relu(z2) # ReLU activation function 2nd hidden layer. output of 2nd hidden layer
        #print ("h2",h2, h2.size())
        

        w3h2 = torch.matmul(self.weights[2], h2) #output layer (wx)
        z3 = torch.add(w3h2, self.biases[2]) #hidden layer1  (wx + bias)     
        out = stable_softmax(z3) # Softmax activation function. output of output layer
        #print ("out",out, out.size())
        return z3, out
    

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        #print ("y_1hot_t_train", y_1hot_t_train, y_1hot_t_train.size())
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t_train) #given

        act_outputs_t = act_outputs.t()
        
        loss = cross_entropy(act_outputs_t, y_1hot) ###ok too
        #loss = (act_outputs_t - y_1hot).pow(2).sum()/act_outputs_t.size(0) #MSE
        #print ("loss1", loss, loss.size())       

        # backward
        loss.backward()
        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            #print ("w.grad.data", w.grad.data)
            #print ("w.data before", w.data)
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()
        #return loss.data[0] # getting the error message
        #Use tensor.item() to convert a 0-dim tensor to a Python number
        return loss.item()
    
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
        outputs, act_outputs = self._feed_forward(X_t)
        act_outputs_t = act_outputs.t()
        #print ("act_outputs_t", act_outputs_t)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t) #given
        
        #loss = cross_entropy(act_outputs, y_1hot[-1].to( dtype= torch.long)) ###ok
        loss = cross_entropy(act_outputs_t, y_1hot)  ##ok too
        #print ("loss2", loss)
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        #print ("hi")
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs, 0)[1]

        #return torch.max(act_outputs[-1], 0)[1] #given
        
###my change
class AutogradNeuralNetworkShallow:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        #print("shape", shape, len(shape)) #shape [784, 10] 2
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.autograd.Variable(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.autograd.Variable(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        self.init_weights()
        #print ("weight ", self.weights)
        #print ("weight init after", self.weights[0])

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        #print (len(self.weights)) # len(self.weights) = 3
        #print ("weight init", self.weights)
        for i in range(len(self.weights)):
            #print (self.weights[i], len(self.weights[i]))
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        w1x = torch.matmul(self.weights[0], X)  #input layer (wx)
        #print ("before bias w1x", w1x, w1x.size())
        z1 = torch.add(w1x, self.biases[0]) # input times first layer matrices   (wx + bias) 
        #print (" after bias", z1, z1.size())   
        #out = F.softmax(z1, dim = 0) # Softmax activation function. output of output layer
        out = stable_softmax(z1)
        #out = softmax(z1)
        #print ("out",out, out.size())
        return z1, out
    

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        #print ("y_1hot_t_train", y_1hot_t_train, y_1hot_t_train.size())
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t_train) #given

        act_outputs_t = act_outputs.t()
        
        loss = cross_entropy(act_outputs_t, y_1hot) ###ok too
        #loss = (act_outputs_t - y_1hot).pow(2).sum()/act_outputs_t.size(0) #MSE
        #print ("loss1", loss, loss.size())       

        # backward
        loss.backward()
        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            #print ("w.grad.data", w.grad.data)
            #print ("w.data before", w.data)
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()
        #return loss.data[0] # getting the error message
        #Use tensor.item() to convert a 0-dim tensor to a Python number
        return loss.item()
    
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
        outputs, act_outputs = self._feed_forward(X_t)
        act_outputs_t = act_outputs.t()
        #print ("act_outputs_t", act_outputs_t)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t) #given
        
        #loss = cross_entropy(act_outputs, y_1hot[-1].to( dtype= torch.long)) ###ok
        loss = cross_entropy(act_outputs_t, y_1hot)  ##ok too
        #print ("loss2", loss)
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        #print ("hi")
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs, 0)[1]

        #return torch.max(act_outputs[-1], 0)[1] #given
class BasicNeuralNetwork:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
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

    def _feed_forward(self, X):
        """Forward pass

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
        w1x = torch.matmul(self.weights[0], X)  #input layer (wx)
        #print ("before bias w1x", w1x, w1x.size())
        z1 = torch.add(w1x, self.biases[0]) # input times first layer matrices   (wx + bias)
        #print ("z1",z1, z1.size())      
        h1 = relu(z1) # ReLU activation function 1st hidden layer. output of 1st hidden layer
        #print ("h1",h1, h1.size())
        
       
        w2h1 = torch.matmul(self.weights[1], h1 ) #hidden layer2 (wx)
        #print ("before bias w2h1", w2h1, w2h1.size())
        z2 = torch.add(w2h1, self.biases[1]) #hidden layer2  (wx + bias)
        h2 = relu(z2) # ReLU activation function 2nd hidden layer. output of 2nd hidden layer
        #print ("h2",h2, h2.size())
        

        w3h2 = torch.matmul(self.weights[2], h2) #output layer (wx)
        #print ("before bias w3h2", w3h2, w3h2.size())
        z3 = torch.add(w3h2, self.biases[2]) #hidden layer1  (wx + bias)     
        out = stable_softmax(z3) # Softmax activation function. output of output layer
        #print ("out",out, out.size())
        return w1x, w2h1, w3h2, h1, h2, out

    def d_relu(self,inp):
    # grad of relu with respect to input activations
        #print (inp)
        inp[inp <= 0.0] = 0.0
        inp[inp > 0.0] = 1.0
        #print (inp)
        return inp

    #def _backpropagation(self, outputs, act_outputs, X, y_1hot):
    def _backpropagation(self, loss, z1, z2, z3, h1, h2, act_outputs_t, X, y_1hot, learning_rate):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """
        
        d_loss = act_outputs_t - y_1hot #error
        #print ("h2", h2, h2.size()) #   h2 = torch.Size([100, 60000])
        #print ("X", X.size(1))
        dw3 = torch.matmul(h2, d_loss)/X.size(1)
        #print ("dw3", dw3, dw3.size()) # torch.Size([100, 10])
        
        #db3 = d_loss
        #print ("db3", db3, db3.size())
        #print ("self.weights[2] inside back prop", self.weights[2])
        dh2 = torch.matmul(d_loss,self.weights[2])
        #print ("dh2", dh2, dh2.size()) #torch.Size([60000, 100])
        #print ("z2", z2.size()) #torch.Size([100, 60000])
        #dh2 = dh2 * self.d_relu(z2.t()) #multiply with derivative of relu(z2)
        dh2 = dh2 * self.d_relu(h2.t()) #multiply with derivative of relu(h2)
        #dh2 =  self.d_relu(dh2)
        #print ("dh2 after", dh2, dh2.size()) #torch.Size([60000, 100])
        dw2 = torch.matmul(h1, dh2) /X.size(1)
        #print ("dw2", dw2, dw2.size()) #[300, 100])
        #db2 = dh2       
        #print ("db2", db2, db2.size())
        
        #print ("self.weights[1] inside back prop", self.weights[1])
        dh1 = torch.matmul(dh2,self.weights[1]) 
        #print ("dh1", dh1, dh1.size()) #torch.Size([60000, 300])
        #dh1 = dh1 * self.d_relu(z1.t()) #multiply with derivative of relu(z1)
        dh1 = dh1 * self.d_relu(h1.t()) #multiply with derivative of relu(z1)
        #dh1 = self.d_relu(dh1)
        #print ("dh1 after", dh1, dh1.size()) #torch.Size([60000, 300)
        #print ("X", X, X.size())

        dw1 = torch.matmul(X, dh1) / X.size(1)
        return dw1, dw2, dw3


    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        #raise NotImplementedError
        #print ("X begin", X, X.size())
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        #print ("y_1hot_t_train", y_1hot_t_train, y_1hot_t_train.size())
        #print ("X_t_train begin", X_t_train)
        # feed forward
        z1, z2, z3, h1, h2, act_outputs = self._feed_forward(X_t_train)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t_train) #given

        act_outputs_t = act_outputs.t()

        #print ("act_outputs_t inside ff", act_outputs_t)
        
        loss = cross_entropy(act_outputs_t, y_1hot) ###ok too
        
        # backpropagation
        dw1, dw2, dw3 = self._backpropagation(loss, z1, z2, z3, h1, h2, act_outputs_t, X_t_train, y_1hot, learning_rate)
        
        self.weights[0] -=  (learning_rate ) * dw1.t()
        #print ("weights[0] after", self.weights[0], self.weights[0].size())
        self.weights[1] -= (learning_rate ) * dw2.t()
        #print ("weights[1] after", self.weights[1], self.weights[1].size())
        self.weights[2] -= (learning_rate) * dw3.t()
        
        self.biases[0] -= learning_rate * 1
        #print ("biases", self.biases[0].size())
        self.biases[1] -= learning_rate * 1
        self.biases[2] -=  learning_rate * 1 

        return loss.item() 

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
        z1, z2, z3, h1, h2, act_outputs = self._feed_forward(X_t)
        act_outputs_t = act_outputs.t()
        #print ("act_outputs_t inside loss", act_outputs_t)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t) #given
        
        #loss = cross_entropy(act_outputs, y_1hot[-1].to( dtype= torch.long)) ###ok
        loss = cross_entropy(act_outputs_t, y_1hot)  ##ok too
        return loss.item()
        

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        '''
        ### given
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]
        '''
        z1, z2, z3, h1, h2, act_outputs = self._feed_forward(X.t())
        
        return torch.max(act_outputs, 0)[1]

        #return torch.max(act_outputs[-1], 0)[1] #given
class BasicNeuralNetworkShallow:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
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

    def _feed_forward(self, X):
        """Forward pass

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
        w1x = torch.matmul(self.weights[0], X)  #input layer (wx)
        #print ("before bias wx1", self.wx1, self.wx1.size())
        z1 = torch.add(w1x, self.biases[0]) # input times first layer matrices   (wx + bias)    
        out = stable_softmax(z1) # Softmax activation function. output of output layer
        #print ("out",out, out.size())
        return z1, out

    def d_relu(self,inp):
    # grad of relu with respect to input activations
        #print (inp)
        inp[inp <= 0.0] = 0.0
        inp[inp > 0.0] = 1.0
        #print (inp)
        return inp

    #def _backpropagation(self, outputs, act_outputs, X, y_1hot):
    def _backpropagation(self, z1, act_outputs_t, X, y_1hot):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """
        
        d_loss = act_outputs_t - y_1hot #error
        #print ("d_loss", d_loss, d_loss.size()) # torch.Size([250, 10])
        #print ("X", X, X.size()) # torch.Size([784, 250])
        dw1 = torch.matmul(X, d_loss)/X.size(1)
        #print ("dw1", dw1.size()) # torch.Size([250, 784])
        #print ("self.weights[0]", self.weights[0].size()) # torch.Size([10, 784])

        return dw1


    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        #raise NotImplementedError
        #print ("X begin", X, X.size())
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        #print ("y_1hot_t_train", y_1hot_t_train, y_1hot_t_train.size())
        #print ("X_t_train begin", X_t_train)
        # feed forward
        z1, act_outputs = self._feed_forward(X_t_train)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t_train) #given

        act_outputs_t = act_outputs.t()

        #print ("act_outputs_t inside ff", act_outputs_t)
        
        loss = cross_entropy(act_outputs_t, y_1hot) ###ok too
        
        # backpropagation
        dw1 = self._backpropagation( z1, act_outputs_t, X_t_train, y_1hot)
        
        self.weights[0] -=  (learning_rate ) * dw1.t()
        self.biases[0] -= learning_rate * 1

        return loss.item() 

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
        z1, act_outputs = self._feed_forward(X_t)
        act_outputs_t = act_outputs.t()
        #print ("act_outputs_t inside loss", act_outputs_t)
        #loss = cross_entropy(act_outputs[-1], y_1hot_t) #given
        
        #loss = cross_entropy(act_outputs, y_1hot[-1].to( dtype= torch.long)) ###ok
        loss = cross_entropy(act_outputs_t, y_1hot)  ##ok too
        return loss.item()
        

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        '''
        ### given
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]
        '''
        z1, act_outputs = self._feed_forward(X.t())
        
        return torch.max(act_outputs, 0)[1]

        #return torch.max(act_outputs[-1], 0)[1] #given     