import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100) # kernel_size = 5*5, stride = 1
        self.fc2 = nn.Linear(100, 10)
        
        #my change for kernel_size = 3*3, stride = 3
        '''
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride = 3, padding = 1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride = 3, padding = 1)
        self.fc1 = nn.Linear(20, 100) #kernel_size = 3*3, stride = 3
        self.fc2 = nn.Linear(100, 10)
        '''

        #my change for kernel_size = 14*14, stride = 1
        '''
        self.conv1 = nn.Conv2d(1, 10, kernel_size=14, stride = 1,  padding = 3) #changed the padding
        self.conv2 = nn.Conv2d(10, 20, kernel_size=14, stride = 1, padding = 3)
        self.fc1 = nn.Linear(20, 100) 
        self.fc2 = nn.Linear(100, 10)
        '''

    def forward(self, x):
        # 2D convolutional layer with max pooling layer and reLU activations
        #x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        # 2nd layer of 2D convolutional layer with max pooling layer and reLU activations
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # rearrange output from a 2d array (matrix) to a vector
        #print(x.shape) ##previously it was torch.Size([32, 20, 4, 4])  kernel_size = 5*5, stride = 1
        #print(x.shape) #torch.Size([32, 20, 1, 1] kernel_size = 3*3, stride = 3
        #print(x.shape) #torch.Size([32, 20, 1, 1] kernel_size = 14*14, stride = 1
        x = x.view(-1, 320) #kernel_size = 5*5, stride = 1
        #x = x.view(-1, 20) #my change kernel_size = 3*3, stride = 3 and kernel_size = 14*14, stride = 1
        # fully connected layer with ReLU activation
        x = F.relu(self.fc1(x)) 
        # fully connected layer
        x = self.fc2(x)
        # log(softmax(x)) activation
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, epoch, printout=False):
    # Initializes model for training
    model.train()
    #Initialize local variables
    loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # move data from CPU to GPU if GPU is available
        data, target = data.to(device), target.to(device) 
        
        # Forward pass. Gets output *before* softmax
        output = model(data)
        
        # Computes the negative log-likelihood over the training data target. 
        #     log(softmax(.)) avoids computing the normalization factor. 
        #     Note that target does not need to be 1-hot encoded (pytorch will 1-hot encode for you)
        #np.random.shuffle(target) #for random shuffling of target
        loss = F.nll_loss(output, target)
        #loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        # why does getting the max of the log-probability suffices for prediction? Don't we need to compute the softmax probability?
        pred = output.max(1, keepdim=True)[1] 
        # Computes accuracy
        correct += pred.eq(target.view_as(pred)).sum().item()
        # **Very important,** need to zero the gradient buffer before we can use the computed gradients
        model.zero_grad()     # zero the gradient buffers of all parameters
        # Backpropagate loss for all paramters over all examples
        loss.backward()
        # Performs one step of SGD with a fixed learning rate (not a Robbins-Monro procedure)
        learning_rate = 0.05
        # iterate over all model parameters
        for f in model.parameters():
            # why subtract? Are we minimizing or maximizing?
            f.data.sub_(f.grad.data * learning_rate)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss.mean(), correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    
    return (100. * correct / len(train_loader.dataset))



def validation(model, device, validation_loader, quiet = False):
    
    # Some operations (such as Dropout and BatchNorm) will behave differently between training and testing.
    #   In our model it does not matter, but it is generally good practice to explain what you intend to do to pytorch
    model.eval()
    
    # Initialize local variables
    validation_loss = 0
    correct = 0
    
    # no_grad() tells pytorch not to worry about keeping intermediate resutls around as we are not looking to 
    #   backpropagate this forward pass
    with torch.no_grad():
        for data, target in validation_loader:
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            
            # sums up the loss over the entire batch
            validation_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # get the index of the max log-probability
            #    why does getting the max of the log-probability suffices for prediction? Don't we need to compute the softmax probability?
            pred = output.max(1, keepdim=True)[1] 
            
            # Computes accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    if not quiet: 
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)))
    
    return (100. * correct / len(validation_loader.dataset))

def save_plots(train_accs, test_accs):
#def save_plots(test_accs):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(test_accs)
    xs = np.arange(n)

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')

def save_flatness_plots(alpha_v,v_err):
    plt.rcParams.update({'font.size': 15})
    plt.xlabel("Alpha")
    plt.ylabel("Validation Error (%)")
    plt.semilogy(alpha_v,v_err,linestyle='-', marker='o', color='b')
    #plt.show()
    plt.savefig('flatness.png')


def main():
    # Train batch size
    train_batch_size = 32

    # How many times we will go over the entire dataset (keeps same number of gradient steps)
    #no_epochs = 10 # for Task 3: question 1, 2
    no_epochs = 100 # for Task 3: question 3(a), 3(b), 3(c)
    #no_epochs = 3

    # Validation batch size (as large as GPU can support)
    validation_batch_size = 8192

    # Do we have GPUs available?py
    use_cuda = torch.cuda.is_available()
    print(f'Is cuda available? {use_cuda}')

    # Handy way to use GPUs if there are any
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loader parameters
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    
    ######## with shuffling the target label ####

    traindata = datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    #print ("traindata before",traindata.targets, traindata.targets.shape)

    np.random.shuffle(traindata.targets)

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=train_batch_size, shuffle=True, **kwargs)

    #print ("traindata after",traindata.targets, traindata.targets.shape)

    '''  
    ######## without shuffling the target label ####
    # Data loader for training data (will download MNIST data from the Web if data is not cached in the system)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                          batch_size=train_batch_size, shuffle=True, **kwargs)

    '''

    # Data loader for test data (will download MNIST data from the Web if data is not cached in the system)
    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                          batch_size=validation_batch_size, shuffle=True, **kwargs)

    # Instantiates the model. If devide="cuda", the model parameters will be created in the GPU memory.
    model = Net().to(device)


    # Copy initial model parameters
    paramsinitial = model.named_parameters()
    initial_params = {}
    for name, param in paramsinitial:
        initial_params[name] = param.data.clone()

    ##
    ## Train Model
    ##

    train_accs = []
    val_accs = []
    for epoch in range(1, no_epochs + 1):
        # Train for one epoch 
        #train(model, device, train_loader, epoch,printout=False)
        train_acc = train(model, device, train_loader, epoch,printout=False)
        
        train_accs.append(train_acc)
        # Validation of model accuracy so far 
        val_acc = validation(model, device, validation_loader)
        val_accs.append (val_acc)
    ##
    ## End Train Model
    ##


    # Save final model parameters
    paramsfinal = model.named_parameters()
    final_params = {}
    for name, param in paramsfinal:
        final_params[name] = param.data.clone()

    

    ###plot accuracy
    save_plots(train_accs, val_accs)
    
    
    ## Flatness plot
    v_err = []
    alpha_v = []

    # Interpolate model parameters
    for alpha in torch.linspace(0, 1.5, steps=10): 
        alpha = alpha.to(device)
        for name, param in model.named_parameters():
            param.data = (1. - alpha)*initial_params[name].data + alpha *final_params[name].data
        v_err.append(100. - validation(model, device, validation_loader,quiet=True))
        alpha_v.append(alpha)


    #save_flatness_plots(alpha_v, v_err)
    
    print("The End.")


