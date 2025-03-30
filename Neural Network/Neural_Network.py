import torch

"""output_size is the number of neurons of the lth layer and input size is the number of layers in the (l-1)th layer."""



class Layers():
    def __init__(self, input_size = None, output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = torch.randn(input_size, output_size)  # Initialize weights randomly
        self.bias = torch.randn(output_size)  # Initialize bias randomly
        self.activation = torch.zeros(output_size)  # Initialize activation to zero

    def sigmoid(self, z):
        return 1/(1-torch.exp(-z))


    def ff(self, previous_layer_activations):
        self.activation = self.sigmoid(previous_layer_activations @ self.weights.T + self.bias)
        


class NeuralNetwork(Layers):
    def __init__(self, number_of_layers, features, labels, input_size, output_size, neurons_per_layer):
        super.__init__(input_size, output_size)
        self.number_of_layers = number_of_layers
        self.features = features
        self.labels = labels
        self.hidden_layers = [] 
        self.neurons_per_layer = neurons_per_layer
        self.n = features.shape[1]
        self.m = features.shape[0]
        self.all_layers = []
        self.input_layer = None
        self.output_layer = None

    def making_layers(self):
        self.input_layer = Layers(output_size=self.n)
        for i in range(1, self.number_of_layers-1):
            self.hidden_layers.append(Layers(self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
        self.output_layer = Layers(self.neurons_per_layer[-2], self.neurons_per_layer[-1])    
        self.all_layers = [self.input_layer,*(self.hidden_layers), self.output_layer]


    def calculation_of_activations(self):
        for i in range(1, self.number_of_layers):
            self.all_layers[i].ff(self.all_layers[i-1].activations)

    
