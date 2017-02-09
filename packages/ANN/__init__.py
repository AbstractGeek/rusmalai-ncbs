import numpy as np


class FNN:
    ''' This is an artificial neural network class.
    It learns using the backpropagation algorithm, and can classify binary
    as well as multi-class problems. At the moment it can only
    run in batch learning mode.'''

    def __init__(self, numLayers, Input, target, hiddenNeuronList=[], eta=0.1,
                 mode='batch', error_function='quadratic'):
        ''' Initialize an instance of the machine learning class '''
        self.mode = mode
        self.numLayers = numLayers
        self.numHiddenLayers = numLayers - 2
        self.eta = eta
        self.error_function = error_function
        self.__Input__ = np.matrix(Input).T

        self.number_of_features = self.__Input__.shape[0]
        self.number_of_training_points = self.__Input__.shape[1]

        self.__Input__ = np.vstack(
            [self.__Input__, [1]*self.number_of_training_points])  # Add bias
        self.class_labels = set(target)

        self.number_of_classes = len(self.class_labels)
        print("Class labels:{}".format(self.class_labels))
        self.set_target(target)

        if not len(hiddenNeuronList):
            # Should be changed later to something more general
            self.hiddenNeuronList = [self.number_of_features] * \
                self.numHiddenLayers
        else:
            self.hiddenNeuronList = hiddenNeuronList

        self.construct_network()
        print("Network constructed with {} layers, learning rate is {}"
              .format(self.numLayers, self.eta))
        self.connect_layers()
        print("Layers connected")

    # Neural network construction methods
    def construct_network(self):
        ''' Construct the different layers and units of the NN '''
        # Input layer Stuff
        self.input_layer = input_layer(self.number_of_features)

        # Create Hidden Layers
        self.hidden_layers = [
            hidden_layer(self.hiddenNeuronList[i],
                         self.number_of_training_points, self.eta)
            for i in range(self.numHiddenLayers)]

        # Create output layer
        self.output_layer = output_layer(
            self.number_of_classes, self.number_of_training_points, self.eta)

        self.layers = [self.input_layer] + self.hidden_layers + \
            [self.output_layer]

    def connect_layers(self):
        '''Connect layers'''
        # Input layer
        self.hidden_layers[0].connect_layer(self.input_layer)
        # Hidden layers
        for n in range(self.numHiddenLayers-1):
            self.hidden_layers[n+1].connect_layer(self.hidden_layers[n])
        # Output layer
        self.output_layer.connect_layer(self.hidden_layers[-1])

    def set_target(self, target):
        ''' Setting target to the ANN'''
        try:
            np.shape(self.__Input__)[0] == len(target)

            if self.number_of_classes > 2:  # More than binary classification
                self.__target__ = np.zeros(  # Expected output from each neuron
                    (self.number_of_classes, self.number_of_training_points))
                for i, label in enumerate(self.class_labels):
                    for j, t in enumerate(target):
                        if label == t:
                            self.__target__[i, j] = 1
            else:
                self.__target__ = np.zeros((1, self.number_of_training_points))
                self.__target__[0] = target

        except:
            return "Lengths of input and target don't match"

    # Cost functions for the NN
    def calculateError(self, t, o):
        '''This is the main error/cost function'''
        if self.error_function == 'quadratic':
            return self.quadratic(t, o)

    def quadratic(self, t, o):
        ''' This is quadratic cost function '''
        return (1./2)*(np.sum(np.square(t-o)))

    # The learning rule and weights updates ##
    def backpropagate(self, target):
        ''' Backpropagation of errors through the NN '''
        self.output_layer.backpropagate(target)
        for layer in self.hidden_layers[::-1]:
            layer.backpropagate()

    def update_weights(self):
        ''' NN weight updates '''
        for layer in self.layers[1:]:
            layer.update()

    # Prediction related methods ####

    def compute_forward(self, input):
        '''Forward computation by NN by passing through activation function'''
        self.input_layer.compute_layer(input)
        for layer in self.hidden_layers:
            layer.compute_layer()
        self.pred_class = self.output_layer.compute_layer()

    def train(self, iterations=1):
        ''' This is the main iteration function which forward computes,
            backpropagates, and updates weights for the NN '''
        error = []
        for i in range(iterations):
            self.compute_forward(self.__Input__)
            self.backpropagate(self.__target__)
            self.update_weights()
            error.append(
                         self.calculateError(
                                             self.__target__,
                                             self.output_layer.output))
            #if i % (iterations/10.) == 0.:
            #    print("{} iterations, loss = {}".format(i+1, error[-1]))
        if iterations == 1:
            return self.output_layer.output, error[0]
        else:
            return self.output_layer.output, error

    def test(self, test_data):
        ''' This is the main function which forward computes
            and classifies test data '''
        self.compute_forward(test_data)
        return self.pred_class


class neuron_layer:
    ''' This is a neural network layer class'''

    def __init__(self, N, numDataPoints, eta):
        ''' This initializes a neural network layer '''
        if isinstance(self, hidden_layer):
            self.N = N+1   # Adding bias neurons to the hidden layers
        else:
            if N == 2:  # Special provision for binary classification
                self.N = 1
            else:
                self.N = N
        self.neurons = [neuron(self, index) for index in range(self.N)]
        self.eta = eta
        self.output = np.zeros((self.N, numDataPoints))
        self.delta = np.zeros((self.N, numDataPoints))

    def connect_layer(self, prev_layer):
        ''' This connects neural network layers together '''
        self.prev_layer = prev_layer
        self.index = self.prev_layer.index + 1
        prev_layer.set_next_layer(self)
        for n in self.neurons:
            n.initialize_weights(prev_layer.N)

    def compute_layer(self):
        ''' Compute activation for all neurons in layer '''
        for i, n in enumerate(self.neurons):
            self.output[i] = n.compute()
            n.set_w_out()  # Setting output weights
        return self.output

    def update(self):
        ''' Update weights for all neurons in layer '''
        for i, neuron in enumerate(self.neurons):
            neuron.change_weight(self.eta)


class input_layer(neuron_layer):
    ''' This is the input layer class'''

    def __init__(self, N):
        self.N = N + 1
        self.index = 0

    def compute_layer(self, x):
        self.output = x
        return self.output

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer


class hidden_layer(neuron_layer):
    ''' This is the hidden layer class'''

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def backpropagate(self):
        next_delta = self.next_layer.delta
        # print neuron.w_out, next_delta
        for i, neuron in enumerate(self.neurons):
            self.delta[i] = neuron.set_delta(neuron.d_activation *
                                             np.dot(neuron.w_out, next_delta))


class output_layer(neuron_layer):
    ''' This is the output layer class'''

    def backpropagate(self, target):
        for i, neuron in enumerate(self.neurons):
            self.delta[i] = neuron.set_delta(
                                              (target[i] - neuron.output) *
                                              neuron.d_activation)


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
        elif self.activation_method == 'step':
            return self.step(input)

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

    def step(self, x, theta=0):
        ''' This is sigmoid activation function. '''
        return 1 if x >= theta else -1

    # Hyperbolic tan activation #
    def tanh(self, x):
        ''' This is tan hyperbolic activation function. '''
        return (2./(1+np.exp(-2*x))) - 1

    def tanh_diff(self, output):
        ''' This is derivative of tan hyperbolic activation function. '''
        return 1 - (output)**2

'''
########### Experimental ###########
'''


class RNN:
    ''' This is the class for Recursive neural nets '''
    pass


class CNN:
    ''' This is the class for Convolutional neural nets '''
    pass
