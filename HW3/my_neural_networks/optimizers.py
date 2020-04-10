import logging
import torch


class BaseOptimizer(object):
    """Optimzier Abstract Class
    """

    def __init__(self, init_learning_rate):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
        """
        self.learning_rate = init_learning_rate
    
    def update(self, weights, biases, W_grads, b_grads, iteration, l2_lambda, m ):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        raise NotImplementedError

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


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent Optimizer
    """

    def __init__(self, init_learning_rate):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
        """
        super(SGDOptimizer, self).__init__(init_learning_rate)

    def update(self, weights, biases, W_grads, b_grads, iteration, l2_lambda, m):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        weights = [w - ((w *l2_lambda * self.learning_rate)/m) -  (g * self.learning_rate)
                    for w, g in zip(weights, W_grads)]
        biases = [b - (self.learning_rate * g.unsqueeze(1))
                    for b, g in zip(biases, b_grads)]
        return weights, biases


class MomentumOptimizer(BaseOptimizer):
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
        #raise NotImplementedError

    def update(self, weights, biases, W_grads, b_grads, iteration, l2_lambda, m):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        rho = 0.9
        
        v_t = []
        v_t_b =[]
        for weight in weights:
            v_t.append(torch.zeros_like(weight))

        for bias in biases:
            v_t_b.append(torch.zeros_like(bias))
        
        #print ("v_t",  v_t[1], v_t[0].size(), v_t[1].size(), v_t[2].size())
        #print ("W_grads", W_grads[1], W_grads[0].size(), W_grads[1].size(), W_grads[2].size())
        #print ("weights", weights[1], weights[0].size(), weights[1].size(), weights[2].size())

        #print ("biases", biases[0].size(), biases[1].size(), biases[2].size())
        #print ("b_grads", b_grads[0].size(), b_grads[1].size(), b_grads[2].size())



       
        #v_new = ([rho * v for v in v_t] + [(self.learning_rate * g) for g in W_grads])
        v_new = [(rho * v) + (self.learning_rate * g) for g, v in zip(W_grads, v_t)]
        #print ("v_new", v_new[1], v_new[0].size(), v_new[1].size(), v_new[2].size())
        
        weights = [w - v for w , v in zip(weights, v_new)]
        #print ("weights",  weights[1], weights[0].size(), weights[1].size(), weights[2].size())

        #v_new_b = ([rho * v for v in v_t_b] + [(self.learning_rate * g.unsqueeze(1)) for g in b_grads]) 
        v_new_b = [(rho * v) + (self.learning_rate * g.unsqueeze(1)) for g, v in zip(b_grads,v_t_b)]
        #print ("v_new", v_new_b[0].size(), v_new_b[1].size(), v_new_b[2].size())
        
        biases = [b - v for b , v in zip(biases, v_new_b)]

        return weights, biases
        
        #raise NotImplementedError


class NesterovOptimizer(BaseOptimizer):
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
        super(NesterovOptimizer, self).__init__(init_learning_rate)
        #raise NotImplementedError

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
        rho = 0.9
        
        v_t = []
        v_t_b =[]
        for weight in weights:
            v_t.append(torch.zeros_like(weight))

        for bias in biases:
            v_t_b.append(torch.zeros_like(bias))

        weights = [(rho * v) + w for w, v in zip(weights, v_t)]
    
        #print ("weights",  weights[1], weights[0].size(), weights[1].size(), weights[2].size())

        biases = [(rho * v) + b for b, v in zip(biases,v_t_b)]
        #print ("biases", biases[0].size(), biases[1].size(), biases[2].size())
        return weights , biases
        

        #raise NotImplementedError

    def update(self, weights, biases, W_grads, b_grads, iteration, l2_lambda, m):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        rho = 0.9
        
        v_t = []
        v_t_b =[]
        for weight in weights:
            v_t.append(torch.zeros_like(weight))

        for bias in biases:
            v_t_b.append(torch.zeros_like(bias))

        v_new = [(rho * v) - (self.learning_rate * g) for g, v in zip(W_grads, v_t)]
        #print ("v_new", v_new[1], v_new[0].size(), v_new[1].size(), v_new[2].size())
        
        weights = [w + v for w , v in zip(weights, v_new)]
        #print ("weights",  weights[1], weights[0].size(), weights[1].size(), weights[2].size())

        #v_new_b = ([rho * v for v in v_t_b] + [(self.learning_rate * g.unsqueeze(1)) for g in b_grads]) 
        v_new_b = [(rho * v) - (self.learning_rate * g.unsqueeze(1)) for g, v in zip(b_grads,v_t_b)]
        #print ("v_new", v_new_b[0].size(), v_new_b[1].size(), v_new_b[2].size())
        
        biases = [b + v for b , v in zip(biases, v_new_b)]

        return weights, biases
        #raise NotImplementedError


class AdamOptimizer(BaseOptimizer):
    """Adam Optimizer

    """
    def __init__(self, init_learning_rate, shape,
                 gpu_id=-1, beta1=0.9, beta2=0.999, epsilon=1e-5): #my change epsilon
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
        #raise NotImplementedError

    def update(self, weights, biases, W_grads, b_grads, iteration, l2_lambda, m):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        beta1=0.9 
        beta2=0.999
        epsilon=1e-5


        m1 = []
        m1_b = []
        m2 =[]
        m2_b = []
        for weight in weights:
            m1.append(torch.zeros_like(weight))

        for bias in biases:
            m1_b.append(torch.zeros_like(bias))

        for weight in weights:
            m2.append(torch.zeros_like(weight))

        for bias in biases:
            m2_b.append(torch.zeros_like(bias))

        # m1 and m2 are the first and second moments

        #m1 = β1 ∗ m1 + (1 − β1)*grad(f(x_t))

        m1 = [(beta1 * mom1 ) + ((1 - beta1) * g ) for g, mom1 in zip(W_grads, m1)]

        m1_b = [(beta1 * mom1 ) + ((1 - beta1) * g.unsqueeze(1) ) for g, mom1 in zip(b_grads, m1_b)]

        #m2 = β2 ∗ m2 + (1 − β2)(grad(f(x_t))^2)

        m2 = [(beta2 * mom2 ) + ((1 - beta2) * g * g ) for g, mom2 in zip(W_grads, m2)]

        m2_b = [(beta2 * mom2 ) + ((1 - beta2) * g.unsqueeze(1) * g.unsqueeze(1) ) for g, mom2 in zip(b_grads, m2_b)]
        
        #u1 and u2 are the first and second moments’ bias correction (iteration t)

        u1 = [mom1 / (1 - beta1 ** iteration ) for mom1 in m1]
        u1_b = [mom1 / (1 - beta1 ** iteration ) for mom1 in m1_b]

        u2 = [mom2 / (1 - beta2 ** iteration) for mom2 in m2]
        u2_b = [mom2 / (1 - beta2 ** iteration) for mom2 in m2_b]

        ### update weights

        div = [(self.learning_rate * uc1) / (torch.sqrt(uc2) + epsilon) for uc1, uc2 in zip(u1,u2)]
        div_b = [(self.learning_rate * uc1) / (torch.sqrt(uc2) + epsilon) for uc1, uc2 in zip(u1_b,u2_b)]

        weights = [w - v for w , v in zip(weights, div)]
        biases = [b - v for b , v in zip(biases, div_b)]

        return weights, biases



        #raise NotImplementedError
