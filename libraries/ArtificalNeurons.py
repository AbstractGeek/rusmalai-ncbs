import numpy as np


class neuron:
    '''This is a neuron (Units inside a layer) class'''

    def __init__(self, layer, index,
                 activation_method='sigmoid', bias_constant=0.99):
        ''' Initialize a neuron instance '''
        self.layer = layer
        self.index = index
        self.activation_method = activation_method
        self.bias_constant = bias_constant

    def initialize_weights(self, numInputs):
        ''' Randomly assign initial weights from a uniform distribution '''
        self.w = np.random.uniform(-1, 1, numInputs)
        # self.w = np.zeros(numInputs) # Just for kicks

    def set_w_out(self):
        ''' Get all weights going out of the neuron '''
        if isinstance(self.layer, output_layer):
            self.w_out = None
        elif isinstance(self.layer, hidden_layer):
            w_out = [n.w[self.index] for n in self.layer.next_layer.neurons]
            self.w_out = np.array(w_out)

    def compute(self):
        ''' Compute the activation output for regular and bias neurons '''
        if not (isinstance(self.layer, hidden_layer) and self.index == 0):
            input = np.ravel(np.dot(np.transpose(self.w),
                             self.layer.prev_layer.output))
            self.output = self.activation(input)
            self.d_activation = self.activation_diff(self.output)
        else:
            factor = self.bias_constant
            # Bias units outputing constants all the time.
            self.output = np.ones(self.layer.prev_layer.output.shape[1]) \
                * factor
            self.d_activation = self.activation_diff(self.output)
        return self.output

    def set_delta(self, delta):
        self.delta = delta
        return self.delta

    def change_weight(self, eta):
        ''' Update weights for neuron '''
        # Seems to work right. Check this once.
        self.w += eta * np.ravel(np.dot(self.delta,
                                        self.layer.prev_layer.output.T))

    # Activation functions #
    def activation(self, input):
        ''' This is our activation function. '''
        if self.activation_method == 'sigmoid':
            return self.sigmoid(input)
        elif self.activation_method == 'tanh':
            return self.tanh(input)

    def activation_diff(self, x):
        ''' This is our activation derivative function. '''
        if self.activation_method == 'sigmoid':
            return self.sigmoid_diff(x)
        elif self.activation_method == 'tanh':
            return self.tanh_diff(x)

    # Sigmoid activation #
    def sigmoid(self, x):
        ''' This is sigmoid activation function. '''
        return 1/(1+np.exp(-x))

    def sigmoid_diff(self, output):
        ''' This is derivative of the sigmoid activation function. '''
        return output*(1-output)

    # Hyperbolic tan activation #
    def tanh(self, x):
        ''' This is tan hyperbolic activation function. '''
        return (2./(1+np.exp(-2*x))) - 1

    def tanh_diff(self, output):
        ''' This is derivative of tan hyperbolic activation function. '''
        return 1 - (output)**2
