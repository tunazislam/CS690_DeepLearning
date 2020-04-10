import logging
import torch


class SGDOptimizer(object):
    """Stochastic Gradient Descent Optimizer
    """

    def __init__(self, init_learning_rate):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
        """
        self.learning_rate = init_learning_rate

    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        weights = [w - (self.learning_rate * g)
                    for w, g in zip(weights, W_grads)]
        biases = [b - (self.learning_rate * g.unsqueeze(1))
                    for b, g in zip(biases, b_grads)]
        return weights, biases

    def ahead(self, weights, biases):
        """Look ahead for weight updates

            This is for NesterovOptimizer, which requires to look ahead for future updates. For other optimizers, this just simply return the inputs.

        Args:
            weights: weights before the udpate
            biases: biases before the update

        Returns:
            weights and biases
        """
        return weights, biases
    
    def back(self, weights, biases):
        return weights, biases


class MomentumOptimizer(SGDOptimizer):
    """Momentum Optimizer

    """
    
    def __init__(self, init_learning_rate, shape, gpu_id=-1, rho=0.9):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            rho: momentum hyperparameter
        """
        super(MomentumOptimizer, self).__init__(init_learning_rate)
        self.gpu_id = gpu_id
        self.shape = shape
        self.rho = rho
        self.reset()

    def reset(self):
        if self.gpu_id == -1:
            self.W_velocity = [torch.FloatTensor(j, i).zero_()
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_velocity = [torch.FloatTensor(i, 1).zero_()
                           for i in self.shape[1:]]
        else:
            self.W_velocity = [torch.randn(j, i).zero_().cuda(self.gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_velocity = [torch.randn(i, 1).zero_().cuda(self.gpu_id)
                           for i in self.shape[1:]]

    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        # update velocity
        self.W_velocity = [self.rho * wv + wg
                            for wv, wg in zip(self.W_velocity, W_grads)]
        self.b_velocity = [self.rho * bv + bg.unsqueeze(1)
                            for bv, bg in zip(self.b_velocity, b_grads)]
        # update weights
        weights = [w - (self.learning_rate * wv)
                    for w, wv in zip(weights, self.W_velocity)]
        biases = [b - (self.learning_rate * bv)
                    for b, bv in zip(biases, self.b_velocity)]
        return weights, biases


class NesterovOptimizer(MomentumOptimizer):
    """Nesterov Optimizer

    """
    def __init__(self, init_learning_rate, shape, gpu_id=-1, rho=0.9):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            rho: momentum hyperparameter
        """
        super(NesterovOptimizer, self).__init__(init_learning_rate, shape, gpu_id, rho)
    
    def ahead(self, weights, biases):
        """Look ahead for weight updates

            This is for NesterovOptimizer, which looks ahead for future updates. For other optimizers, this simply returns the inputs.

            Think about how Nesterov Momentum requires for calculating gradients.

        Args:
            weights: weights before the udpate
            biases: biases before the update

        Returns:
            weights and biases
        """
        weights = [w + self.rho * wv
                    for w, wv in zip(weights, self.W_velocity)]
        biases = [b + self.rho * bv
                    for b, bv in zip(biases, self.b_velocity)]
        return weights, biases
    
    def back(self, weights, biases):
        weights = [w - self.rho * wv
                    for w, wv in zip(weights, self.W_velocity)]
        biases = [b - self.rho * bv
                    for b, bv in zip(biases, self.b_velocity)]
        return weights, biases

    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        weights, biases = self.back(weights, biases)

        # update velocity
        self.W_velocity = [self.rho * wv - self.learning_rate * wg
                            for wv, wg in zip(self.W_velocity, W_grads)]
        self.b_velocity = [self.rho * bv - self.learning_rate * bg.unsqueeze(1)
                            for bv, bg in zip(self.b_velocity, b_grads)]
        # update weights
        weights = [w + wv
                    for w, wv in zip(weights, self.W_velocity)]
        biases = [b + bv
                    for b, bv in zip(biases, self.b_velocity)]
        return weights, biases


class AdamOptimizer(SGDOptimizer):
    """Adam Optimizer

    """
    def __init__(self, init_learning_rate, shape,
                 gpu_id=-1, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            beta1: adam hyperparameter
            beta2: adam hyperparameter
            epsilon: to avoid divided by zero
        """
        super(AdamOptimizer, self).__init__(init_learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gpu_id = gpu_id
        self.shape = shape
        self.reset()

    def reset(self):
        self.it = 1
        if self.gpu_id == -1:
            self.W_firstm = [torch.FloatTensor(j, i).zero_()
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_firstm = [torch.FloatTensor(i, 1).zero_()
                           for i in self.shape[1:]]
            self.W_secondm = [torch.FloatTensor(j, i).zero_()
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_secondm = [torch.FloatTensor(i, 1).zero_()
                           for i in self.shape[1:]]
        else:
            self.W_firstm = [torch.randn(j, i).zero_().cuda(self.gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_firstm = [torch.randn(i, 1).zero_().cuda(self.gpu_id)
                           for i in self.shape[1:]]
            self.W_secondm = [torch.randn(j, i).zero_().cuda(self.gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.b_secondm = [torch.randn(i, 1).zero_().cuda(self.gpu_id)
                           for i in self.shape[1:]]
    
    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        # update moments
        self.W_firstm = [self.beta1 * wfm + (1 - self.beta1) * wg
                            for wfm, wg in zip(self.W_firstm, W_grads)]
        self.b_firstm = [self.beta1 * bfm + (1 - self.beta1) * bg.unsqueeze(1)
                            for bfm, bg in zip(self.b_firstm, b_grads)]
        self.W_secondm = [self.beta2 * wsm + (1 - self.beta2) * wg * wg
                            for wsm, wg in zip(self.W_secondm, W_grads)]
        self.b_secondm = [self.beta2 * bsm + (1 - self.beta2) * bg.unsqueeze(1) * bg.unsqueeze(1)
                            for bsm, bg in zip(self.b_secondm, b_grads)]
        # unbias
        W_first_unbias = [wfm / (1 - self.beta1 ** self.it)
                            for wfm in self.W_firstm]
        b_first_unbias = [bfm / (1 - self.beta1 ** self.it)
                            for bfm in self.b_firstm]
        W_second_unbias = [wsm / (1 - self.beta2 ** self.it)
                            for wsm in self.W_secondm]
        b_second_unbias = [bsm / (1 - self.beta2 ** self.it)
                            for bsm in self.b_secondm]
        # update weights
        weights = [w - (self.learning_rate * wfu / (torch.sqrt(wsu) + self.epsilon))
                    for w, wfu, wsu in zip(weights, W_first_unbias, W_second_unbias)]
        biases = [b - (self.learning_rate * bfu / (torch.sqrt(bsu) + self.epsilon))
                    for b, bfu, bsu in zip(biases, b_first_unbias, b_second_unbias)]
        self.it += 1
        return weights, biases
