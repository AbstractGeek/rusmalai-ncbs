import ANN

numLayers = 3
input = [[0,0],[0,1],[1,0],[1,1]]
target = [1,0,0,1]
nn1 = ANN.ANN(numLayers, input, target )
e = nn1.iterate(100)
