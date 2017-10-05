import tensorflow as tf

# Neighbors2
# (3,3)
# min row: 1
# max row: 5
# min col: 1
# max col: 5
# (4,3)
# min row: 2
# max row: 6
# min col: 1
# max col: 5
# So it might be a good idea to have 4x4 filters,
# as that would cover all neighs2
#
# Padding with 0's is the natural choice since that would be
# equivalent to having empty cells outside of grid
#
# For a policy network, i.e. with actions [0, 1, ..., n_channels-1]
# corresponding to the probability of assigning the different channels,
# how can the network know, or be trained to know, that some actions
# are illegal/unavailable?


class Net:
    def __init__(self, logger,
                 *args, **kwargs):
        self.logger = logger

    def forward(self, inp):
        """
        Forward pass. Given an input, such as a feature vector
        or the whole state, return the output of the network.
        """
        pass

    def backward(self, loss):
        """
        Backward pass.
        """
        pass

    def save(self, filenam):
        """
        Save parameters to disk
        """
        pass


class RSValNet(Net):
    """
    Input is a grid with the number of used channels for each cell.
    Output is a state value.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RSPolicyNet(Net):
    """
    Input is a grid with the number of used channels for each cell.
    Output is a vector with probability for each channel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
