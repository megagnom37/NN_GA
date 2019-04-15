from network import *
import numpy as np
import itertools

class ConvolutionNetwork:
    def __init__(self):
        self.nn = NNetwork()
        self.filters = []

    def add_filter(self, size):
        self.filters.append(2 * np.random.rand(*size) - 1)
        
    def get_weights_list(self):
        result = []

        for f in self.filters:
            result += list(itertools.chain(*f.tolist()))

        result += self.nn.get_weights_as_list()
        return result

    def set_weights_from_list(self, weights):
        for i in range(len(self.filters)):
            offset = 0
            size = self.filters[i].shape[0] * self.filters[i].shape[1]
            self.filters[i] = np.array(weights[offset:size]).reshape(self.filters[i].shape)
            offset += size
        
        self.nn.set_weights_from_list(weights[offset:])

