import random as rand
import math


class Neuron:
    def __init__(self, num_w):
        self.w = [rand.uniform(-1.0, 1.0) for i in range(num_w + 1)]
        self.a = 0.0

    def activate(self, x):
        result = 0.0
        for i in range(len(self.w)):
            result += self.w[i] * x[i]
        self.a = 1 / (1 + math.e ** (-result))

    def __str__(self):
        result = ''
        if not self.w:
            return str(self.a)
        for i, _w in enumerate(self.w):
            result += 'w%s=%s ' % (i, _w)
        result += 'a=%s' % self.a
        return result
