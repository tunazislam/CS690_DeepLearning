"""© 2018 Jianfei Gao All Rights Reserved"""
import os
import numpy as np
from collections import defaultdict
import torch


class CoraDataset(object):
    def __init__(self, root):
        """Initialize the class

        Args
        ----
        root : Str
            Root folder of cora dataset.

        """
        # allocate data buffer
        self.feat_mx = []
        self.label_mx = []

        self.pt_str2int = {}
        self.label_str2int = {}

        # read feature file
        path = os.path.join(root, 'cora.content')
        file = open(path, 'r')
        content = file.readlines()
        file.close()
        for line in content:
            # parse each line
            if len(line) == 0:
                break
            else:
                entity = line.strip().split()
            pt_str = entity[0]
            label_str = entity[-1]
            feat = np.array([float(itr) for itr in entity[1:-1]])

            # numerical point ID
            if pt_str in self.pt_str2int:
                pass
            else:
                self.pt_str2int[pt_str] = len(self.pt_str2int)
            pt_int = self.pt_str2int[pt_str]

            # numerical label
            if label_str in self.label_str2int:
                pass
            else:
                self.label_str2int[label_str] = len(self.label_str2int)
            label_int = self.label_str2int[label_str]

            self.feat_mx.append(feat)
            self.label_mx.append(label_int)

        # formalize matrix format
        self.feat_mx = np.array(self.feat_mx, dtype=float)
        self.label_mx = np.array(self.label_mx, dtype=int)

        # allocate adjacent matrix based on number of nodes
        num_nodes = len(self.pt_str2int)
        self.adj_mx = np.zeros(shape=(num_nodes, num_nodes), dtype=float)

        # read connection file as undirected graph
        path = os.path.join(root, 'cora.cites')
        file = open(path, 'r')
        content = file.readlines()
        file.close()
        for line in content:
            if len(line) == 0:
                break
            else:
                entity = line.strip().split()
            pt_int1 = self.pt_str2int[entity[0]]
            pt_int2 = self.pt_str2int[entity[1]]
            self.adj_mx[pt_int1, pt_int2] = 1
            self.adj_mx[pt_int2, pt_int1] = 1

        # compute degree of graph
        self.max_deg = int(self.adj_mx.sum(axis=1).max())
        self.avg_deg = self.adj_mx.sum(axis=1).mean()
        #print ("max_deg", self.max_deg) #168

        # make average adjacent matrix
        # >>>>> YOUR CODE STARTS HERE <<<<<
        self.min_deg = int(self.adj_mx.sum(axis=1).min())
        #self.adj_mx = <...>
        #print (self.adj_mx, self.adj_mx.shape, type(self.adj_mx.size)) # (2708, 2708)
        #unique, counts = np.unique(self.adj_mx, return_counts=True)
        #print(dict(zip(unique, counts)))
        #print (self.adj_mx.sum(axis=1), self.adj_mx.sum(axis=1).shape) #(2708,)
        #print (1/self.adj_mx.sum(axis=1), (1/self.adj_mx.sum(axis=1)).shape) #(2708,)
        #print(np.diag(1/self.adj_mx.sum(axis=1)), np.diag(1/self.adj_mx.sum(axis=1)).shape)
        ### sum of adj_mx column-wise 
        self.sum_adj_mx = self.adj_mx.sum(axis=1)
        ###### for avg we need to divide the adj_mx with sum_adj_mx so need to do 1/adj_mx
        self.denom_sum_adj_mx = 1/self.sum_adj_mx
        ###make it diagonal to preverve dimension
        self.diag_adj_mx = np.diag(self.denom_sum_adj_mx) 
        #print(self.diag_adj_mx, self.diag_adj_mx.shape)
        self.adj_mx = self.diag_adj_mx  @ self.adj_mx
        
        # I = np.matrix(np.eye(self.adj_mx.shape[0])) ##create identity matrix to add self-loop
        # self.adj_mx = self.adj_mx + I
        #print (self.adj_mx, self.adj_mx.size,  type(self.adj_mx.size))
        
        
        # >>>>> YOUR CODE ENDS HERE <<<<<

        # put everything in torch version
        self.feat_mx = torch.from_numpy(self.feat_mx).float()
        self.label_mx = torch.from_numpy(self.label_mx).long()
        self.adj_mx = torch.from_numpy(self.adj_mx).float()

        # statistics
        print('=' * 41)
        print("{:25s}".format('CORA Dataset'))
        print('-' * 41)
        print("Feature Matrix  : {:>9s} {:>13s}".format(
                'x'.join([str(dim) for dim in self.feat_mx.size()]),
                str(self.feat_mx.dtype)))
        print("Label Matrix    : {:>9s} {:>13s}".format(
                'x'.join([str(dim) for dim in self.label_mx.size()]),
                str(self.label_mx.dtype)))
        print("Adjacent Matrix : {:>9s} {:>13s}".format(
                'x'.join([str(dim) for dim in self.adj_mx.size()]),
                str(self.adj_mx.dtype)))
        print("Min Degree      : {:>23d}".format(self.min_deg))
        print("Max Degree      : {:>23d}".format(self.max_deg))
        print("Average Degree  : {:>23.3f}".format(self.avg_deg))
        print('=' * 41)
        print()


if __name__ == '__main__':
    dataset = CoraDataset(root='./Data/cora')
