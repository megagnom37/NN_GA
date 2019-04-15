from neuron import *

class NNetwork:
    def __init__(self):
        self.layers = []
        self.output_layer = []

    def set_input_layer(self, neurons_num):
        input_layer = [Neuron(-1) for i in range(neurons_num)]
        self.layers.append(input_layer)

    def set_hide_layer(self, neurons_num):
        hide_layer = [Neuron(len(self.layers[-1])) for i in range(neurons_num)]
        self.layers.append(hide_layer)

    def set_output_layer(self, neurons_num):
        self.set_hide_layer(neurons_num)
        self.output_layer = self.layers[-1]

    def activate_neurons(self, x):
        input_neurons = self._normalization_x(x, norma=1)
        for i_neuron in range(len(self.layers[0])):
            self.layers[0][i_neuron].a = input_neurons[i_neuron]

        for i_layer in range(1, len(self.layers)):
            activate_value_prev_layer = [i.a for i in self.layers[i_layer-1]] + [1.0]
            for i_neuron in range(len(self.layers[i_layer])):
                self.layers[i_layer][i_neuron].activate(activate_value_prev_layer)
        
    def _normalization_x(self, x, norma=100):
        x_tmp = x[:]
        for i in range(len(x_tmp)):
            x_tmp[i] /= norma
        return x_tmp 

    def get_solution(self, x):
        self.activate_neurons(x)
        return [x.a for x in self.layers[-1]]

    def get_weights_as_list(self):
        result = []
        for layer in self.layers[1:]:
            for neuron in layer:
                for weight in neuron.w:
                    result.append(weight)
        return result

    def set_weights_from_list(self, weigths):
        idx = 0
        for layer in self.layers[1:]:
            for neuron in layer:
                for i in range(len(neuron.w)):
                    neuron.w[i] = weigths[idx]
                    idx += 1

    def __str__(self):
        result = ''
        for i, layer in enumerate(self.layers):
            neurons = ''
            for neuron in layer:
                neurons += '<%s>\n' % (neuron)
            result += '==== layer %s ====\n%s' % (i, neurons)
        return result

