import numpy as np
import time

class Weights:
    def __init__(self, layer):
        self.layer = layer.get_weights()
        if len(self.layer) == 0:
            self.w_shape = None
            self.b_shape = None
            self.layer_size = 0
        elif len(self.layer) == 2:
            self.b_shape = self.layer[1].shape
            self.b = self.layer[1].tolist()
            self.w_shape = self.layer[0].shape
            self.w = self.layer[0].reshape((-1,)).tolist()
            self.layer_size = len(self.w) + len(self.b)
        elif len(self.layer) == 3:
            self.w_shape = self.layer[0].shape
            self.w = self.layer[0].reshape((-1,)).tolist()
            self.c_shape = self.layer[1].shape
            self.c = self.layer[1].reshape((-1,)).tolist()
            self.b_shape = self.layer[2].shape
            self.b = self.layer[2].tolist()
            self.layer_size = len(self.w) + len(self.b) + len(self.c)

    
    def size(self):
        return self.layer_size

    def get_weights_list(self):
        if not self.w_shape and not self.b_shape:
            return []
        if len(self.layer) == 2:
            return self.w + self.b
        if len(self.layer) == 3:
            return self.w + self.c + self.b

    def get_weights_mtrx(self, weights):
        if not self.w_shape and not self.b_shape:
            return []
        
        if len(self.layer) == 2:
            self.w = weights[:len(self.w)]
            self.b = weights[len(self.w):]
            w_mtrx = np.array(self.w).reshape(self.w_shape)
            b_mtrx = np.array(self.b).reshape(self.b_shape)
            return [w_mtrx, b_mtrx]
        
        if len(self.layer) == 3:
            self.w = weights[:len(self.w)]
            self.c = weights[len(self.w):len(self.w) + len(self.c)]
            self.b = weights[len(self.w) + len(self.c):]
            w_mtrx = np.array(self.w).reshape(self.w_shape)
            c_mtrx = np.array(self.c).reshape(self.c_shape)
            b_mtrx = np.array(self.b).reshape(self.b_shape)
            return [w_mtrx, c_mtrx, b_mtrx]

