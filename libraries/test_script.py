import ANN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

#input = data.data.T[:2].T
#target = data.target
numLayers = 3 
iterations = 20000

input = [[0,0],[0,1],[1,0],[1,1]]
target = [0,1,1,0]
nn1 = ANN.FNN(numLayers, input, target, eta=0.05 )

e = nn1.train(iterations)
achieved = nn1.output_layer.output
print achieved
#e = []
#for i in range(20000):
#    e.append(nn1.iterate())
    #print "Output is {}".format(nn1.output_layer.output)
    #print nn1.output_layer.neurons[0].w, nn1.output_layer.prev_layer.neurons[0].output, nn1.output_layer.prev_layer.neurons[1].output, nn1.output_layer.prev_layer.neurons[2].output
plt.plot(e)
plt.show()
#plt.scatter(input.T[0], input.T[1], c= 0.5*(target-achieved)**2)
#plt.show()
