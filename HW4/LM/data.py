"""© 2018 Jianfei Gao All Rights Reserved"""
import os
import math
import numpy as np
import torch
import sys

class PTBDataset(object):
    def __init__(self, path, dictionary):
        r"""Initialize the class

        Args
        ----
        root : Str
            Path of a PTB dataset file.
        dictionary : Dict
            Dictionary from words to integers.

        """
        # allocate data buffer
        self.seq = []

        self.str2int = dictionary

        # read feature file
        seq = []
        file = open(path, 'r')
        content = file.readlines()
        file.close()
        for line in content:
            # parse each line
            if len(line) == 0:
                continue
            else:
                sentence = line.strip().split() + ['<eos>']
            for word in sentence:
                if word in self.str2int:
                    pass
                else:
                    self.str2int[word] = len(self.str2int)
                seq.append(self.str2int[word])

        # formalize data sequences
        self.seq = np.array(seq, dtype=int)

        # statistics
        print('=' * 29)
        print("PTB: {:<25s}".format(path))
        print('-' * 29)
        print("Dictionary: {:>17s}".format("@{}".format(id(dictionary))))
        print('# Word IDs: {:>17d}'.format(len(dictionary)))
        print("# Words   : {:>17d}".format(len(self.seq)))
        print('=' * 29)
        print()

class MarkovChainLoader(object):
    def __init__(self, dataset, order):
        r"""Initialize the class

        Args
        ----
        dataset : PTBDataset
            A raw PTB dataset.
        order : Int
            Markov chain order.

        """
        seq = dataset.seq
        #print("seq", seq, len(seq)) #seq [ 0  1  2 ... 39 26 24] 929589
        self.order = order
        #print("order", order) 
        self.input_seq = np.zeros((len(seq) - 1, order), dtype=int)
        #print("input_seq", self.input_seq, self.input_seq.shape) #  (929588, order)
        self.target_seq = np.zeros((len(seq) - 1,), dtype=int)
        #print("target_seq", self.target_seq, self.target_seq.shape)
        for i in range(len(seq) - 1):
            for j in range(0, order):
                if i - j >= 0:
                    token = seq[i - j] #Suppose for the t-th word wt, we have its context ct = [wt−k; wt−k+1; · · · ; wt−1]. 
                else:
                    token = -1 #If the words of context go out of the context, we will just use token -1
                self.input_seq[i, (order - 1) - j] = token #context
            self.target_seq[i] = seq[i + 1]

        # You must populate these variables
        self.counts = {}
        self.sums = {}

        #print(self.input_seq.shape, self.target_seq.shape)
        # (929588, 1) (929588,)
        # (73759, 1) (73759,)
        # (82429, 1) (82429,)
        #print (self.target_seq)
        # [ 1  2  3 ... 39 26 24]
        # [ 396 1129   64 ...  108   27   24]
        # [ 78  54 251 ...  87 214  24]
        for input, target in zip(self.input_seq, self.target_seq):
            input = tuple(input)
            #print(input)
            # >>>>> YOUR CODE STARTS HERE <<<<<
            #<...>
            if input in self.counts:
                if target in self.counts[input]:
                    self.counts[input][target] += 1
                else:
                    self.counts[input][target] = 1
                self.sums[input] += 1
            else:
                self.counts[input] = {target: 1}
                self.sums[input] = 1
            # >>>>> YOUR CODE ENDS HERE <<<<<

class BPTTBatchLoader(object):
    def __init__(self, dataset, bptt, batch_size=20):
        r"""Initialize the class

        Args
        ----
        dataset : PTBDataset
            A raw PTB dataset.
        bptt : Int
            Length of truncated backpropagation through time.
        batch_size : Int
            Batch size.

        """
        # truncate dataset for batch splitting
        seq = dataset.seq
        #print("seq", seq, seq.shape) #train: seq [ 0  1  2 ... 39 26 24] (929589,)
        num_words = len(seq)
        num_batches = num_words // batch_size
        num_words = num_batches * batch_size
        seq = seq[0:num_words]

        self.bptt = bptt

        # >>>>> YOUR CODE STARTS HERE <<<<<
        # divide sequences into batch level
        #self.seq = <...>
        #print(seq, seq.shape) # [   0    1    2 ... 9999  119 1143] (929580,)
        self.seq = seq.reshape(batch_size, -1).T
        #print(self.seq, self.seq.shape, len(self.seq)) 
        # compute number of bptt batches based on fixed bptt
        #self.num_bptt = <...>
        self.num_bptt = math.ceil(len(self.seq) / bptt)
        #print(self.num_bptt) # 9296
        
        # >>>>> YOUR CODE ENDS HERE <<<<<

        # statistics
        print('=' * 29)
        print('BPTT Batch Loader')
        print('-' * 29)
        print("Batch Sequence Length: {:>5d}".format(self.seq.shape[0]))
        print("Batch Size           : {:>5d}".format(self.seq.shape[1]))
        print("BPTT                 : {:>5d}".format(self.bptt))
        print("# BPTT Batches       : {:>5d}".format(self.num_bptt))
        print('=' * 29)
        print()

        #sys.exit()

    def __len__(self):
        r"""Length of the class"""
        return self.num_bptt

    def __iter__(self):
        r"""Iterator of the class"""
        return self.Iterator(self)

    class Iterator(object):
        def __init__(self, loader):
            r"""Initialize the class"""
            self.loader = loader
            self.ptr = 0

        def __len__(self):
            r"""Length of the class"""
            return len(self.loader)

        def __next__(self):
            r"""Next element of the class"""
            # validate next element
            if self.ptr >= self.loader.num_bptt:
                raise StopIteration
            else:
                pass

            # update pointers in raw data for next element
            ptr0 = self.ptr * self.loader.bptt
            ptr1 = min(ptr0 + self.loader.bptt, len(self.loader.seq) - 1)
            self.ptr += 1

            # get input and target batch
            input_batch = self.loader.seq[ptr0:ptr1]
            target_batch = self.loader.seq[ptr0 + 1:ptr1 + 1]
            input_batch = torch.LongTensor(input_batch).contiguous()
            target_batch = torch.LongTensor(target_batch).contiguous()
            return input_batch, target_batch

if __name__ == '__main__':
    dictionary = {}
    train_dataset = PTBDataset('./data/ptb/ptb.train.txt', dictionary)
    valid_dataset = PTBDataset('./data/ptb/ptb.valid.txt', dictionary)
    test_dataset = PTBDataset('./data/ptb/ptb.test.txt', dictionary)
    train_loader = MarkovChainLoader(train_dataset, 3)
    valid_loader = MarkovChainLoader(valid_dataset, 3)
    test_loader = MarkovChainLoader(test_dataset, 3)
    train_loader = BPTTBatchLoader(train_dataset)
    valid_loader = BPTTBatchLoader(valid_dataset)
    test_loader = BPTTBatchLoader(test_dataset, batch_size=1)