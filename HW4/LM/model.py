"""© 2018 Jianfei Gao All Rights Reserved"""
import numpy as np
import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    def __init__(self, num_labels, num_embeds, num_hidden, num_layers, dropout=0.5):
        r"""Initialize the class

        Args
        ----
        num_labels : Int
            Size of input and output labels.
        num_embeds : Int
            Size of word embeddings.
        num_hidden : Int
            Size of hidden layers.
        num_layers : Int
            Number of recurrent layers in the recurrent group.
        dropout : Float
            Dropout rate.

        """
        # parse arguments
        super(LSTMLM, self).__init__()

        # allocate layers
        self.encoder1 = nn.Embedding(num_labels, num_embeds)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(num_embeds, num_hidden, num_layers, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.decoder3 = nn.Linear(num_hidden, num_labels)
        self.init_weights()

        # initialize hidden state
        self.states = None

    def init_weights(self):
        r"""Initialize Weights"""
        initrange = 0.1
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder3.bias.data.zero_()
        self.decoder3.weight.data.uniform_(-initrange, initrange)

    def detach_states(self, states):
        r"""Detach hidden staes

        Args
        ----
        states : None or torch.Tensor or (torch.Tensor, ...)
            Hidden states of LSTM model.

        """
        if states is None:
            return None
        elif isinstance(states, tuple):
            return tuple([self.detach_states(itr) for itr in states])
        elif hasattr(states, 'detach'):
            return states.detach()
        else:
            raise TypeError("cannot detach {}".format(type(states)))

    def forward(self, x):
        r"""Forwarding

        Args
        ----
        x : torch.Tensor
            Input batch.
            It should be of shape (#bptt, #batch, #labels).

        Returns
        -------
        y : torch.Tensor
            Output batch.
            It should be of shape (#bptt, #batch, #labels).

        """
        # detach hidden state
        self.states = self.detach_states(self.states)

        # forwarding
        x = self.encoder1(x)
        x = self.drop1(x)
        x, self.states = self.lstm2(x, self.states)
        x = self.drop2(x)
        flat_x = x.view(x.size(0) * x.size(1), x.size(2))
        flat_y = self.decoder3(flat_x)
        y = flat_y.view(x.size(0), x.size(1), flat_y.size(1))
        return y


if __name__ == '__main__':
    input = torch.Tensor(np.random.randint(0, 2, size=(4, 30)))
    model = MCLM(3, 10, 16, 16)
    model(input)

    input = torch.LongTensor(np.random.randint(0, 10, size=(5, 3)))
    model = LSTMLM(10, 16, 16, 2)
    model(input)
