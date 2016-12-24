import ANN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

input = data.data
target = data.target
print input
print target
numLayers = 10 
iterations = 2000
#input = [[0,0.5,1],[0,1, 0.5],[1,0, 0.5],[1,1,1],[0,1.0,0.5]]
#target = [0,1,1,0.5,1]
nn1 = ANN.ANN(numLayers, input, target, eta=0.05 )

e = nn1.iterate(iterations)
#e = []
#for i in range(20000):
#    e.append(nn1.iterate())
    #print "Output is {}".format(nn1.output_layer.output)
    #print nn1.output_layer.neurons[0].w, nn1.output_layer.prev_layer.neurons[0].output, nn1.output_layer.prev_layer.neurons[1].output, nn1.output_layer.prev_layer.neurons[2].output

plt.plot(e)
plt.show()
