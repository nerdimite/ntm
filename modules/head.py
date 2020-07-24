import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):

    def __init__(self, mode, ctrl_dim, memory_unit_size):
        super(Head, self).__init__()
        # Valid modes are 'r' and 'w' for reading and writing respectively
        self.mode = mode
        # Size of each memory vector (key size)
        self.m = memory_unit_size
        
        self.max_shift = 1
        
        # Linear Layers for converting controller output to addressing parameters
        self.key_fc = nn.Linear(ctrl_dim, self.m)
        self.key_strength_fc = nn.Linear(ctrl_dim, 1)
        self.interpolation_gate_fc = nn.Linear(ctrl_dim, 1)
        self.shift_weighting_fc = nn.Linear(ctrl_dim, 3)
        self.sharpen_factor_fc = nn.Linear(ctrl_dim, 1)
        self.erase_weight_fc = nn.Linear(ctrl_dim, self.m)
        self.add_data_fc = nn.Linear(ctrl_dim, self.m)
        
        # Reset
        self.reset()
        
    def forward(self, ctrl_state, prev_weights, memory):
        '''Extracts the parameters and returns the attention weights
        Args:
            ctrl_state (tensor): output vector from the controller (batch_size, ctrl_dim)
            prev_weights (tensor): previous attention weights (batch_size, N)
            memory (nn.Module): memory module
        '''
        
        # Extract the parameters from controller state
        key = torch.tanh(self.key_fc(ctrl_state))
        beta = F.softplus(self.key_strength_fc(ctrl_state))
        gate = torch.sigmoid(self.interpolation_gate_fc(ctrl_state))
        shift = F.softmax(self.shift_weighting_fc(ctrl_state), dim=1)
        gamma = 1 + F.softplus(self.sharpen_factor_fc(ctrl_state))
        erase = torch.sigmoid(self.erase_weight_fc(ctrl_state))
        add = torch.tanh(self.add_data_fc(ctrl_state))
        
        # ==== Addressing ====
        # Content-based addressing
        content_weights = memory.content_addressing(key, beta)
        
        # Location-based addressing 
        # Interpolation
        gated_weights = self._gated_interpolation(content_weights, prev_weights, gate)
        # Convolution
        shifted_weights = self._conv_shift(gated_weights, shift)
        # Sharpening
        weights = self._sharpen(shifted_weights, gamma)
        
        # ==== Read / Write Operation ====
        # Read
        if self.mode == 'r':
            read_vec = memory.read(weights)
        # Write
        elif self.mode == 'w':
            memory.write(weights, erase, add)
            read_vec = None
        else:
            raise ValueError("mode must be read ('r') or write ('w')")
        
        return weights, read_vec
    
    def _gated_interpolation(self, w, prev_w, g):
        '''Returns the interpolated weights between current and previous step's weights
        Args:
            w (tensor): weights (batch_size, N)
            prev_w (tensor): weights of previous timestep (batch_size, N)
            g (tensor): a scalar interpolation gate (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        return (g * w) + ((1 - g) * prev_w)
    
    def _conv_shift(self, w, s):
        '''Returns the convolved weights
        Args:
            w (tensor): weights (batch_size, N)
            s (tensor): shift weights (batch_size, 2 * max_shift + 1)
        Returns:
            (tensor): convolved weights (batch_size, N)
        '''
        batch_size = w.size(0)
        max_shift = int((s.size(1) - 1) / 2)
        
        unrolled = torch.cat([w[:, -max_shift:], w, w[:, :max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1), s.unsqueeze(1))[range(batch_size), range(batch_size)]
    
    def _sharpen(self, w, gamma):
        '''Returns the sharpened weights
        Args:
            w (tensor): weights (batch_size, N)
            gamma (tensor): gamma value for sharpening (batch_size, 1)
        Returns:
            (tensor): sharpened weights (batch_size, N)
        '''
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)
        
    def reset(self):
        '''Reset/initialize the head parameters'''
        # Weights
        nn.init.xavier_uniform_(self.key_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.key_strength_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.interpolation_gate_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.shift_weighting_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.sharpen_factor_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.add_data_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.erase_weight_fc.weight, gain=1.4)
        
        # Biases
        nn.init.normal_(self.key_fc.bias, std=0.01)
        nn.init.normal_(self.key_strength_fc.bias, std=0.01)
        nn.init.normal_(self.interpolation_gate_fc.bias, std=0.01)
        nn.init.normal_(self.shift_weighting_fc.bias, std=0.01)
        nn.init.normal_(self.sharpen_factor_fc.bias, std=0.01)
        nn.init.normal_(self.add_data_fc.bias, std=0.01)
        nn.init.normal_(self.erase_weight_fc.bias, std=0.01)
        