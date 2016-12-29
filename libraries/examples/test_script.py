import ANN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()

input = data.data
#.T[:2].T
target = data.target
numLayers = 3
iterations = 20000

# input = [[0,0],[0,1],[1,0],[1,1]]
# target = [0,1,1,0]
nn1 = ANN.FNN(numLayers, input, target, eta=0.005)

#output, error = nn1.train(iterations)
target = nn1.__target__

error = []
output = []
out, e = nn1.train()
error.append(e)
output.append(out)

plt.ion()

f, ax = plt.subplots(1,2)

im = ax[0].imshow(target, interpolation = 'none', cmap='viridis', origin='lower', aspect='auto', vmin= 0., vmax = 1.)
ax[0].set_yticks([0,1,2])
ax[0].set_yticklabels(['0', '1', '2'])
ax[0].set_ylabel("Classes")
ax[0].set_xlabel("Data Points")
ax[0].set_title('Target classes')

f.canvas.draw()
f.colorbar(im, ax=ax[0])
plt.pause(2)

im = ax[0].imshow(out, interpolation = 'none', cmap='viridis', origin='lower', aspect='auto', vmin= 0., vmax = 1.)

cost = ax[1].plot(error, c='k')
ax[1].set_title("Change in cost Function")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Cost Function")
f.canvas.draw()

for i in range(iterations):
    out, e = nn1.train()
    output.append(out)
    error.append(e)
    #print("Output is {}".format(nn1.output_layer.output))
    #print(nn1.output_layer.neurons[0].w,
    #      nn1.output_layer.prev_layer.neurons[0].output,
    #      nn1.output_layer.prev_layer.neurons[1].output,
    #      nn1.output_layer.prev_layer.neurons[2].output)
    if i % 10 == 0: # Every 10th iteration
        im.set_data(out) 
        ax[1].plot(error, c='k')
        f.canvas.draw()
        plt.pause(0.0001)

plt.ioff()

plt.plot(error)
plt.title("Cost function with iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
