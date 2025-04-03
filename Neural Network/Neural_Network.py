import torch

"""output_size is the number of neurons of the lth layer and input size is the number of layers in the (l-1)th layer."""

def relu(z, derivative = False):
        if derivative:
            return (z>0).float()
        return z if z>0 else 0

def mse(pred, target, derivative = False):
    if derivative:
        return 2 * (pred - target) / target.size(0)
    return ((pred - target) ** 2).mean()

class Layers():
    def __init__(self, input_size = None, output_size=None):
        self.weights = torch.randn(input_size, output_size) * 0.01  # Initialize weights randomly
        self.bias = torch.randn(output_size)  # Initialize bias randomly
        self.activation = torch.zeros(output_size)  # Initialize activation to zero

        def forward(self, previous_layer_activations):
            self.activation = relu(previous_layer_activations @ self.weights.T + self.bias)
            return self.activations
        

class NeuralNetwork():
    def __init__(self, features, labels):
        self.layers = []
        
    def add(self, layer): # layer is an object of the Layer class, initialize it then pass it in.
        self.layers.append(layer)
    
    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x) # for every iteration, x resets to a new value sent after calculation in forward function of Layers class
        return x 





model = NeuralNetwork(features=[1,2,3,4,4], labels=[1,0,1,1,0])
model.add(Layers(2,4))
model.add(Layers(3,2))
























