"""© 2018 Jianfei Gao All Rights Reserved"""
import math
import numpy as np
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    r"""Graph Convolution Layer of Mean Aggregation

    For a given node $v$, its neighbors $N_v$ and features $x_v$, the embedding of aggregation
    will be
    $$
    h_v = W \left[ x_v, \frac{1}{|N_v|} \sum\limits_{u \in N_v}{x_u} \right] + b
    $$

    """
    def __init__(self, num_feats, num_embeds):
        r"""Initialize the class

        Args
        ----
        num_feats : Int
            Size of input features for each node.
        num_embeds : Int
            Size of output embeddings for each node.
        num_samples : Int
            Number of neighbor sampling.

        """
        # parse arguments
        super(GraphConvolution, self).__init__()

        # allocate weight and bias
        self.weight = nn.Parameter(torch.Tensor(num_feats * 2, num_embeds))
        self.bias = nn.Parameter(torch.Tensor(1, num_embeds))
        self.init_weights()

    def init_weights(self):
        r"""Initialize Weights"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        else:
            pass
        #print(self.weight.shape) #torch.Size([2866, 16]), torch.Size([32, 7])

    def forward(self, adj_mx, feats):
        r"""Forwarding

        Args
        ----
        adj_mx : torch.Tensor
            Adjacent matrix of the graph.
            It should be of shape (#nodes, #nodes).
            This is the matrix computed when loading the data
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
            This will be

        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of all nodes.
            It should be of shape (#nodes, #embeddings).

        """
        # This function computes the output of the convolutional layer,
        # before applying the activation (the activation is done by the
        # next class).

        # >>>>> YOUR CODE STARTS HERE <<<<<
        #Z = <...>
         # I = np.matrix(np.eye(adj_mx.shape[0])) ##create identity matrix to add self-loop
        # adj_mx = adj_mx + I
        # #print("adj_mx+identity", adj_mx, adj_mx.shape) #torch.Size([2708, 2708])
        # Ax =  torch.mm(adj_mx.float() , feats)
        # #print("Ax", Ax, Ax.shape) #torch.Size([2708, 1433])
        # ########Normalizing feature representat

        # D_hat = torch.sum(adj_mx, dim=0) #degree of each node
        # print(D_hat, D_hat.shape) #torch.Size([2708])
        # D_hat = torch.diag(D_hat) #convert degree of each node to diagonal matrix
        # print(D_hat, D_hat.shape) #torch.Size([2708, 2708])
        # D_inv = torch.inverse(D_hat)
        # print(D_inv, D_inv.shape)
        # Z = torch.matmul(D_inv.float(), Ax)
        # print(Z,Z.shape) #torch.Size([2708, 1433])

        #D_hat = np.array(np.sum(A_hat, axis=0))[0]
        #D_hat = np.matrix(np.diag(D_hat))

        # z = D_hat**-1 * adj_mx * feats * self.weight + self.bias
        # print(z)
        # print("weight", self.weight, self.weight.shape) #torch.Size([2866, 16])
        # print("bias", self.bias, self.bias.shape) #torch.Size([1, 16])
        # print("feats", feats, feats.shape) #torch.Size([2708, 1433])
       
        Ax =  torch.mm(adj_mx , feats)
        #print("Ax", Ax, Ax.shape) #torch.Size([2708, 1433])
        Ax = torch.cat([feats, Ax], dim=1)
        #print("Ax", Ax, Ax.shape) #torch.Size([2708, 2866])
        Z =  torch.mm(Ax , self.weight) + self.bias
        #print(Z, Z.shape) #torch.Size([2708, 16])
        # >>>>> YOUR CODE ENDS HERE <<<<<
        return Z


class VariableGCN(nn.Module):
    def __init__(self, num_feats, num_hidden, num_labels, num_layers=10,
                 num_samples=5, dropout=0.5):
        r"""Initialize the class

        Args
        ----
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_labels : Int
            Size of output label distribution for each node.
        num_layers : Int
            Number of graph convolutional layers.
        num_samples : Int
            Number of neighbor sampling for LSTM aggregator.
        dropout : Int
            Dropout rate.

        """
        # parse arguments
        super(VariableGCN, self).__init__()

        # allocate layers
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        layers = []
        for i in range(num_layers - 2):
            layers.append(GraphConvolution(num_hidden, num_hidden))
        layers = [GraphConvolution(num_feats, num_hidden)] + layers
        layers = layers + [GraphConvolution(num_hidden, num_labels)]
        self.layers = nn.ModuleList(layers)

    def forward(self, adj_mx, feats):
        r"""Forwarding

        Args
        ----
        adj_mx : torch.Tensor
            Adjacent matrix of the graph.
            It should be of shape (#nodes, #nodes).
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).

        Returns
        -------
        probs : torch.Tensor
            Output label distribution matrix of all nodes before softmax.
            It should be of shape (#nodes, #labels).

        """
        embeds = self.layers[0](adj_mx, feats)
        embeds = self.act(embeds)
        embeds = nn.functional.normalize(embeds, p=2, dim=1)
        for layer in self.layers[1:-1]:
            embeds = layer(adj_mx, embeds)
            embeds = self.act(embeds)
            embeds = nn.functional.normalize(embeds, p=2, dim=1)
        probs = self.layers[-1](adj_mx, embeds)
        self.embeds = embeds

        return probs

