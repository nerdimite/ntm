import torch
import torch.nn.functional as F
from torch import nn

class Memory(nn.Module):

    def __init__(self, memory_units, memory_unit_size):
        super(Memory, self).__init__()
        
        # N = No. of memory units (rows)
        self.n = memory_units
        # M = Size of each memory unit (cols)
        self.m = memory_unit_size
        
        # Define the memory matrix of shape (batch_size, N, M)
        self.memory = torch.zeros([1, self.n, self.m])
        
        # Layer to learn initial values for memory reset
        # self.memory_bias_fc = nn.Linear(1, self.n * self.m)
        
        # Reset/Initialize
        self.reset()
        
    def read(self, weights):
        '''Returns a read vector using the attention weights
        Args:
            weights (tensor): attention weights (batch_size, N)
        Returns:
            (tensor): read vector (batch_size, M)
        '''
        read_vec = torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)
        return read_vec
    
    def write(self, weights, erase_vec, add_vec):
        '''Erases and Writes a new memory matrix
        Args:
            weights (tensor): attention weights (batch_size, N)
            erase_vec (tensor): erase vector (batch_size, M)
            add_vec (tensor): add vector (batch_size, M)
        '''
        # Erase
        memory_erased = self.memory * (1 - weights.unsqueeze(2) * erase_vec.unsqueeze(1))
        # Add
        self.memory = memory_erased + (weights.unsqueeze(2) * add_vec.unsqueeze(1))
        
    def content_addressing(self, key, beta):
        '''Performs content addressing and returns the content_weights
        Args:
            key (tensor): key vector (batch_size, M)
            beta (tensor): key strength scalar (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        # Compare key with every location in memory using cosine similarity
        similarity_scores = F.cosine_similarity(key.unsqueeze(1), self.memory, dim=2)
        # Apply softmax over the product of beta and similarity scores
        content_weights = F.softmax(beta * similarity_scores, dim=1)
        
        return content_weights

    def reset(self, batch_size=1):
        '''Reset/initialize the memory'''
        # Parametric Initialization
        # in_data = torch.tensor([[0.]]) # dummy input
        # Generate initial memory values
        # memory_bias = torch.sigmoid(self.memory_bias_fc(in_data))
        # Push it to memory matrix
        # self.memory = memory_bias.view(self.n, self.m).repeat(batch_size, 1, 1)
        
        # Uniform Initialization of 1e-6
        self.memory = torch.Tensor().new_full((1, self.n, self.m), 1e-6)