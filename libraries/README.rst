ANN
--------

Basics
======
This is a package for running and experimenting with Artificial Neural Networks.

To create a neural network, simply do::

    >>> import ANN 
    >>> nn1 = ANN.FNN(total_number_of_layers, input, target)

This will generate an instance of a neural network with total_number_of_layers. To train the NN to classify:: 

    >>> output, error = nn1.train(number_of_iterations)

This runs the neural network for number_of_iterations and the error vector and the last output is given as the output.
