import ANN
import matplotlib.pyplot as plt

numLayers = 3 
input = [[0,0],[0,1],[1,0],[1,1]]
target = [1,0,0,1]
nn1 = ANN.ANN(numLayers, input, target, eta=0.4 )
e = nn1.iterate(100)

print e
plt.plot(e)
plt.show()
