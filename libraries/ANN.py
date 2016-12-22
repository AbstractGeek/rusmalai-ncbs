import numpy as np


class ANN:
    ''' This is an artificial neural network'''

    def __init__(self, numLayers, Input, target, hiddenNeuronList=[], eta=0.1):
        self.numLayers = numLayers
        self.numHiddenLayers = numLayers - 2
        self.eta = eta
        self.__Input__ = np.matrix(Input)
        self.number_of_features = self.__Input__.shape[1]
        self.set_target(target)

        if not len(hiddenNeuronList):
            # Should be changed later to something more general
            self.hiddenNeuronList =
            [self.number_of_features]*self.numHiddenLayers
        else:
            self.hiddenNeuronList = hiddenNeuronList

        self.construct_network()
        self.connect_layers()

    def set_target(self, target):
        ''' Setting target to the ANN'''
        try:
            np.shape(self.__Input__)[1] == len(target)
            self.target = np.array(target)
        except:
            return "Lengths of input and target don't match"

    def construct_network(self):
        # Input layer Stuff
        self.input_layer = input_layer(self.__Input__)

        # Create Hidden Layers
        self.hidden_layers = [hidden_layer(self.hiddenNeuronList[i], self.eta)
                              for i in range(self.numHiddenLayers)]

        # Create output layer
        self.output_layer = output_layer(1, self.eta)

        self.layers = [self.input_layer] +
        self.hidden_layers + [self.output_layer]

    def connect_layers(self):
        '''Connect layers'''
        # Input layer
        self.hidden_layers[0].connect_layer(self.input_layer)
        # Hidden layers
        for n in range(self.numHiddenLayers-1):
            self.hidden_layers[n+1].connect_layer(self.hidden_layers[n])
        # Output layer
        self.output_layer.connect_layer(self.hidden_layers[-1])

    def __error_function__(self, t, o):
        '''This is the error function'''

        return 1/2*(np.sum(np.square(t-o)))

    def backpropagate(self, target):
        self.output_layer.backpropagate(target)
        for layer in self.hidden_layers[::-1]:
            layer.backpropagate()

    def update_weights(self):
        for layer in self.layers[1:]:
            layer.update()

    def compute_forward(self):
        self.input_layer.compute_layer(self.__Input__)
        for layer in self.hidden_layers:
            layer.compute_layer()
        self.output_layer.compute_layer()

    def iterate(self, iterations):
        error = []
        for i in range(iterations):
            self.compute_forward()
            self.backpropagate(self.target)
            self.update_weights()
            error.append(self.__error_function__(
                self.target, np.array(self.output_layer.output)))
        return error


class neuron_layer:
    ''' This is a neural network layer'''

    def __init__(self, N, eta):
        self.N = N
        self.neurons = [neuron(self) for i in range(N)]
        self.eta = eta

    def connect_layer(self, prev_layer):
        self.prev_layer = prev_layer
        prev_layer.set_next_layer(self)
        numEdges = prev_layer.N * self.N
        for n in self.neurons:
            n.initialize_weights(prev_layer.N)

    def compute_layer(self):
        self.output = [n.compute(self.prev_layer.output) for n in self.neurons]
        return self.output


class input_layer(neuron_layer):
    ''' This is the input layer'''

    def __init__(self, Input):
        self.N = Input.shape[1]
        self.output = self.compute_layer(Input)

    def compute_layer(self, x):
        return np.array(x)

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer


class hidden_layer(neuron_layer):

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def backpropagate(self):
        self.delta = []
        next_delta = self.next_layer.delta
        for i, neuron in enumerate(self.neurons):
            w_next = []
            for n in self.next_layer.neurons:
                w_next.append(n.w[i])
            print np.shape(next_delta), np.shape(w_next)
            self.delta.append(neuron.set_delta(
                neuron.output * (1 - neuron.output) *
                np.dot(np.array(w_next), next_delta)))

    def update(self):
        for i, neuron in enumerate(self.neurons):
            neuron.change_weight(self.eta)


class output_layer(neuron_layer):

    def backpropagate(self, target):
        self.delta = [neuron.set_delta(
            (target[i] - neuron.output) * neuron.output * (1 - neuron.output))
            for i, neuron in enumerate(self.neurons)]

    def update(self):
        for i, neuron in enumerate(self.neurons):
            neuron.change_weight(self.eta)


class neuron:
    '''Units inside a layer'''

    def __init__(self, layer):
        self.layer = layer

    def initialize_weights(self, numInputs):
        self.w = np.random.uniform(-1, 1, numInputs)
        # self.w = np.zeros(numInputs) # Just for kicks

    def sigmoid(self, x):
        ''' This is our activation function. '''
        return 1/(1+np.exp(-x))

    def compute(self, x):
        self.output = self.sigmoid(np.dot(self.w, x))
        return self.output

    def set_delta(self, delta):
        self.delta = delta
        return self.delta

    def change_weight(self, eta):
        print np.shape(self.delta), np.shape(self.layer.prev_layer.output)
        self.w += eta * self.delta * np.array(self.layer.prev_layer.output)

    # def get_delta(target):
    # delta = (self.output - target) * self.output * (1 - self.output)
