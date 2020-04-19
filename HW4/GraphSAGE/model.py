"""© 2018 Jianfei Gao All Rights Reserved"""
import math
import numpy as np
import torch
import torch.nn as nn


class GCN2(nn.Module):
    def __init__(self, aggregator, num_feats, num_hidden, num_labels,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_labels : Int
            Size of output labels.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(
            aggregator, None, num_feats, num_hidden, act, dropout)
        self.gc2 = GraphConvolution(
            aggregator, self.gc1, num_hidden, num_labels, None, dropout)

    def forward(self, adj_list, feats, batch):
        r"""

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        probs : torch.Tensor
            Output label distribution matrix of all nodes before softmax.
            It should be of shape (#nodes, #labels).

        """
        return self.gc2(adj_list, feats, batch)

class GCN6(nn.Module):
    def __init__(self, aggregator, num_feats, num_hidden, num_labels,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_labels : Int
            Size of output labels.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        super(GCN6, self).__init__()

        self.gc1 = GraphConvolution(
            aggregator, None, num_feats, num_hidden, act, dropout)
        self.gc2 = GraphConvolution(
            aggregator, self.gc1, num_hidden, num_hidden, act, dropout)
        self.gc3 = GraphConvolution(
            aggregator, self.gc2, num_hidden, num_hidden, act, dropout)
        self.gc4 = GraphConvolution(
            aggregator, self.gc3, num_hidden, num_hidden, act, dropout)
        self.gc5 = GraphConvolution(
            aggregator, self.gc4, num_hidden, num_hidden, act, dropout)
        self.gc6 = GraphConvolution(
            aggregator, self.gc5, num_hidden, num_labels, None, dropout)

    def forward(self, adj_list, feats, batch):
        r"""

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        probs : torch.Tensor
            Output label distribution matrix of all nodes before softmax.
            It should be of shape (#nodes, #labels).

        """
        return self.gc6(adj_list, feats, batch)

class MeanAggregator(nn.Module):
    def forward(self, feats):
        r"""Forwarding

        Args
        ----
        feats : torch.Tensor
            Features of all nodes to aggregate.
            It should be of shape (#nodes, #features).
        
        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of aggregation.
            It should be of shape (#nodes, #embeddings).

        """

        # >>>>> YOUR CODE STARTS HERE <<<<<
        #embeds = <...>
        embeds = feats.mean(dim = 0)
        #print (embeds)
        # >>>>> YOUR CODE ENDS HERE <<<<<

        return embeds

class LSTMAggregator(nn.Module):
    def __init__(self, num_feats, num_embeds):
        r"""Initialize the class

        Args
        ----
        num_feats : Int
            Size of input features for nodes to aggregate.
        num_embeds : Int
            Size of output embeddings of node aggregation.

        """
        super(LSTMAggregator, self).__init__()

        # allocate LSTM layer
        self.lstm = nn.LSTM(num_feats, num_embeds, batch_first=True)

    def forward(self, feats):
        r"""Forwarding

        Args
        ----
        feats : torch.Tensor
            Features of all nodes to aggregate.
            It should be of shape (#nodes, #features).
        
        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of aggregation.
            It should be of shape (#nodes, #embeddings).

        """
        # >>>>> YOUR CODE STARTS HERE <<<<<
        #embeds = <...>
        #print("view",feats.view(feats.size(0), 1, feats.size(1))) #network dimension
        #print("feats.size(1)", feats.size(1))
        _, (embeds, _) = self.lstm(feats.view(feats.size(0), 1, feats.size(1)), None)

        #print("embeds", embeds, embeds.shape) # torch.Size([1, 2, 1433])
        embeds_sqz = embeds[-1]
        #print("embeds sqz", embeds_sqz, embeds_sqz.shape) # torch.Size([2, 1433])
        return embeds_sqz[-1]
        # >>>>> YOUR CODE ENDS HERE <<<<<

        #return embeds

class JanossyPool(nn.Module):
    def __init__(self, module, k=None):
        r"""Initialize the class

        Args
        ----
        module : nn.Module
            Part of neural network to apply Janossy pooling.
        k : Int
            If using k-ary Janossy Pooling, this is the value of k to use

        """
        super(JanossyPool, self).__init__()
        self.module = module
        self.num_samples = 1
        self.k = k

    def set_num_samples(self, n):
        self.num_samples = n

    def forward(self, x):
        r"""Forwarding

        Args
        ----
        x : torch.Tensor
            Input tensor which supports permutation.

        Returns
        -------
        :x  torch.Tensor
            Output tensor which is permutation invariant of input.

        """
        indices = np.arange(x.size(0))

        total = 0
        for i in range(self.num_samples):
            # Get new permutation
            np.random.shuffle(indices)  #randomly sampled permutation

            # Permute and keep only first k values
            permuted_input = x[indices]
            k_ary_input = permuted_input[:self.k]

            # Pass through the permutation sensitive function and accumulate
            total += self.module(k_ary_input)
        total = total / self.num_samples
        return total

class GraphConvolution(nn.Module):
    def __init__(self, aggregator, features, num_feats, num_hidden,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        features : nn.Module or Func
            Neural network layer which provides previous aggregated embeddings.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_samples : Int
            Number of neighbor sampling for LSTM aggregator.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        # parse arguments
        super(GraphConvolution, self).__init__()
        self.features = features
        self.act_mode = act

        # allocate layers
        if aggregator == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator == 'LSTM':
            self.aggregator = JanossyPool(LSTMAggregator(num_feats, num_feats), k=5)
            self.aggregator.set_num_samples(1)
        else:
            raise ValueError('unsupported \'aggregator\' argument')
        self.fc = nn.Linear(num_feats * 2, num_hidden)
        if self.act_mode is None:
            pass
        elif self.act_mode == 'relu':
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError('unsupported \'act\' argument')

    def forward(self, adj_list, feats, batch):
        r"""Forwarding

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of all nodes.
            It should be of shape (#nodes, #embeddings).

        """
        # aggregate neighbor features
        neighbor_embeds = []

        for i, node in enumerate(batch):

            # get neighbors of previous layer
            neighbors = list(adj_list[node])

            # This will get the embedding of the neighbors, either from
            # the original node features or from the output of the
            # previous layers
            if self.features is None:
                neighbor_feats = feats[neighbors]
            else:
                neighbor_feats = self.features(adj_list, feats, neighbors)

            # Aggregate embedding of neighbors
            # >>>>> YOUR CODE STARTS HERE <<<<<
            #h_neighbors = <...>
            h_neighbors = self.aggregator(neighbor_feats)
            #print("h_neighbors", h_neighbors, h_neighbors.shape) #torch.Size([1433])

            # >>>>>  YOUR CODE ENDS HERE <<<<<

            # Store the aggragated features
            neighbor_embeds.append(h_neighbors)

        #print("neighbor_embeds", neighbor_embeds, len(neighbor_embeds)) #4

        # Make a tensor of (#nodes, #features)
        neighbor_embeds = torch.stack(neighbor_embeds)

        # get the embedding of the nodes (either their original features
        # or the output of previous layer)
        if self.features is None:
            node_feats = feats[batch]
        else:
            node_feats = self.features(adj_list, feats, batch)

        # >>>>> YOUR CODE STARTS HERE <<<<<
        #z = <...>
        z = torch.cat([node_feats, neighbor_embeds], dim=1) #concatenate node features and neighbor embedding
        #print("z",z, z.shape) #torch.Size([2708, 16])
        # >>>>> YOUR CODE ENDS HERE <<<<<

        # Apply the activation and dropout or just return the compute z
        if self.act_mode is None:
            embeds = self.fc(z)
        else:
            embeds = self.drop(self.act(self.fc(z)))

        return embeds

