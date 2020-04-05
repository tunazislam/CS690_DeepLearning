import numpy as np
import torch

EPSILON = 1e-14

def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs predicted label
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels True label

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    #print ("y_1hot true:", y_1hot, y_1hot.size()) #torch.Size([10, 60000])
    #print ("X predicted", X, X.size()) #  torch.Size([410]
    #ce = torch.sum(- y_1hot * torch.log(X + epsilon) )
    X_clamp = torch.clamp(X, epsilon, 1-epsilon)
    #print ("X_clamp", X_clamp)
    
    examples = X.size()[0]            # number of examples
    #print ("examples", examples, type(examples))
    #X_log = torch.log(X)   # compute the log of softmax values
    #print ("X_log", X_log, X_log.size())
    
    reverted = torch.argmax(y_1hot, dim=1) #convert 1-hot encoding to actual label
    #print ("reverted", reverted, reverted[0], reverted.size())  
    #outputs = X_log[range(examples), reverted] # pick the values corresponding to the labels
    #print ("outputs",outputs, outputs.size(0))
    #ce = -torch.sum(outputs)/examples

    log_like = -torch.log(X_clamp[range(examples), reverted])
    #print ("log_like", log_like, log_like.size())
    #return (eX.transpose(0, 1) / eX.sum(dim=1)).transpose(0, 1)
    ce = log_like.sum()/examples
    print ("ce loss", ce)
    return ce
    #raise NotImplementedError



def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    #X_exp = torch.exp(X)
    #X_exp_sum = torch.sum(X_exp)
    #return X_exp / X_exp_sum
    maxes = torch.max(X, 1, keepdim=True)[0]
    #print ("maxes", maxes, maxes.size())
    x_exp = torch.exp(X-maxes)
    #print ("x_exp", x_exp, x_exp.size())
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    #print ("x_exp_sum", x_exp_sum, x_exp_sum.size())
    output_custom = x_exp/x_exp_sum
    #print ("output_custom", output_custom, output_custom.size)
    return output_custom
   
    #print ("eX", eX)
    #print ("soft ", eX / eX.sum(dim=0))
    #eX = torch.exp(X)
    #return (eX / eX.sum(dim=0))



def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    #print (torch.max(X))
    
    X_exp = torch.exp(X - torch.max(X, 0, keepdim=True)[0])
    #X_exp_sum = torch.sum(X_exp)
    #return X_exp / X_exp_sum
    return (X_exp / torch.sum(X_exp, 0, keepdim=True))
    
    #raise NotImplementedError


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    return torch.max(torch.zeros_like(X),X)
    #raise NotImplementedError


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)


If we have one round R =1:
    finding the maximum will take O(n^2) time
If we have 2 round protocol:
    R =2:
        
for round 1: we will divide them into k groups
    we have k groups and each group has n/k elements.
    Find the maximum of each group will take (n/k)^2 times
    So for k group;s it will thake: k* (n/k)^2 times = n^2/k
        
for 2nd round: we will find maxima of group maximas. we have k elements here. It will take O(k^2) times
    
Total: n^2/k + k^2......eq(1)
     
To equalize our work for round 1 and round 2:
    k^2 = n^2/k
    so k = n^(2/3)
    
now putting the value of k in eqn (1) we get:
    
    total number of comparisons= 2* n^(4/3) = O(n^(4/3))
    
    
def find_max(a,n):
    result = []
    for i in range (0, n):
        max = 0
        for j in range (len(a)):
            if a[j] > max:
                max = a[j]
        a.remove(max)
        result.append(max)
    return result
