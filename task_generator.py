# =====================
# Original Source: https://github.com/vlgiitr/ntm-pytorch/tree/master/ntm/datasets
# =====================

import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial

class CopyDataset(Dataset):
    """A Dataset class to generate random examples for the copy task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params):
        '''Initialize a dataset instance for copy task.

        Args:
            task_params (dict): a dict containing parameters relevant to copy task.
        '''
        self.seq_width = task_params['seq_width']
        self.min_seq_len = task_params['min_seq_len']
        self.max_seq_len = task_params['max_seq_len']

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # Get sequence length
        seq_len = torch.randint(self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long).item()
        # Generate sequences
        prob = 0.5 * torch.ones([seq_len, self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()

        # Fill in input sequence, two bit longer and wider than target
        input_seq = torch.zeros([seq_len + 2, self.seq_width + 2])
        input_seq[0, self.seq_width] = 1.0  # start delimiter
        input_seq[1:seq_len + 1, :self.seq_width] = seq
        input_seq[seq_len + 1, self.seq_width + 1] = 1.0  # end delimiter
        
        # Create target sequence
        target_seq = torch.zeros([seq_len, self.seq_width])
        target_seq[:seq_len, :self.seq_width] = seq
        
        return {'input': input_seq, 'target': target_seq}


class AssociativeDataset(Dataset):
    """A Dataset class to generate random examples for associative recall task.

    Each input consists of a list of items with the number of itmes between
    `min_item` and `max_item`. An item is a sequence of binary vectors bounded
    on left and right by delimiter symbols. The list is followed by query item
    selected randomly from input items. It too is bounded by query delimiters.

    Target returns the item next to the query item.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for Associative Recall task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to associative recall task.
        """
        self.seq_width = task_params["seq_width"]
        self.seq_len = task_params["seq_len"]
        self.min_item = task_params["min_item"]
        self.max_item = task_params["max_item"]

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        num_item = torch.randint(self.min_item, self.max_item, (1,), dtype=torch.long).item()
        prob = 0.5 * torch.ones([self.seq_len, self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob)

        # fill in input two bit wider than target to account for delimiter
        # flags.
        input_items = torch.zeros([(self.seq_len + 1) * (num_item + 1) + 1, self.seq_width + 2])
        for i in range(num_item):
            input_items[(self.seq_len + 1) * i, self.seq_width] = 1.0
            input_items[(self.seq_len + 1) * i + 1:(self.seq_len + 1) * (i + 1), :self.seq_width] = seq.sample()

        # generate query item randomly
        # in case of only one item, torch.randint throws error as num_item-1=0
        query_item = 0
        if num_item != 1:
            query_item = torch.randint(
                0, num_item - 1, (1,), dtype=torch.long).item()
        query_seq = input_items[(self.seq_len + 1) * query_item +
                                1:(self.seq_len + 1) * (query_item + 1), :self.seq_width]
        input_items[(self.seq_len + 1) * num_item,
                    self.seq_width + 1] = 1.0  # query delimiter
        input_items[(self.seq_len + 1) * num_item + 1:(self.seq_len + 1)
                    * (num_item + 1), :self.seq_width] = query_seq
        input_items[(self.seq_len + 1) * (num_item + 1),
                    self.seq_width + 1] = 1.0  # query delimiter

        # generate target sequences(item next to query in the input list)
        target_item = torch.zeros([self.seq_len, self.seq_width])
        # in case of last item, target sequence is zero
        if query_item != num_item - 1:
            target_item[:self.seq_len, :self.seq_width] = input_items[
                (self.seq_len + 1) * (query_item + 1) + 1:(self.seq_len + 1) * (query_item + 2), :self.seq_width]

        return {'input': input_items, 'target': target_item}

