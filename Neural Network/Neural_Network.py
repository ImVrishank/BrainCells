import torch

# hyperparameters
epochs = 1000

def relu(z, derivative = False):
    if derivative:
        return (z>0).float()
    return torch.maximum(z, torch.tensor(0.0))

def mse(pred, target, derivative = False):
    if derivative:
        return 2 * (pred - target) / target.size(0)
    return ((pred - target) ** 2).mean()



class Layers():
    def __init__(self, input_size = None, output_size=None):
        self.weights = torch.randn(output_size, input_size) * 0.01  # Initialize weights randomly
        self.bias = torch.randn(1, output_size)  # Initialize bias randomly
        self.activation = torch.zeros(output_size, 1)  # Initialize activation to zero
        self.z = torch.zeros(output_size, 1) 
        self.error = torch.zeros(output_size, 1)

    def forward(self, inputs): 
        # inputs is the previous layer's activations 
        self.z = torch.matmul(inputs, self.weights.T) + self.bias
        self.activation = relu(self.z) # this is the output of the layer
        return self.activation
        
    def calc_backpropagation(self, next_layer_errors = None, next_layer_weights = None, last_layer = False, Y = None):
        # calculating the error of the layer's neurons (must be called every training example)
        if last_layer:
            self.error = (self.activation - Y) * relu(self.z, derivative=True)
        else:
            self.error = (next_layer_errors @ next_layer_weights) * relu(self.z, derivative=True)
        
        return self.error, self.weights
    
    def SGD(self, inputs, lr):
        grad_w = self.error.T @ inputs
        grad_b = self.error.sum(dim=0,keepdim=True)

        self.weights -= lr * grad_w # SGD formula
        self.bias -= lr * grad_b # SGD formula
       

class NeuralNetwork():
    def __init__(self, features, labels, learning_rate):
        self.layers = []
        self.X_train = features
        self.Y_train = labels
        self.cost = 0
        self.learning_rate = learning_rate

        
    def add(self, layer): # layer is an object of the Layer class, initialize it then pass it in.
        self.layers.append(layer)
    
    def feed_forward(self, X):
        # X as the parameter to this function is the features of that training example
        for layer in self.layers:
            X = layer.forward(X) # for every iteration, X resets to a new value sent after calculation in forward function of Layers class
            
        return X
    
    def backpropagation(self, Y):
        self.layers[-1].calc_backpropagation(last_layer = True, Y=Y)

        for i in range(len(self.layers)-2,-1,-1):
            next_layer = self.layers[i+1]
            self.layers[i].calc_backpropagation(next_layer_errors=next_layer.error,
                                                next_layer_weights=next_layer.weights)

        for i in range(len(self.layers)):
            input_ = self.X_train if i == 0 else self.layers[i - 1].activation
            self.layers[i].SGD(input_, self.learning_rate)

        



# train the model
features = torch.tensor([[1.0], [2.0], [3.0], [4.0], [4.0]])
labels = torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0]])

# Initialize the network.
model = NeuralNetwork(features=features,
                        labels=labels,
                        learning_rate=0.01)

model.add(Layers(1, 4))
model.add(Layers(4, 1))

# Training loop:
for epoch in range(epochs):
    # Forward pass over the entire batch.
    output = model.feed_forward(model.X_train)
    cost = mse(output, model.Y_train)
    
    # Perform backpropagation and update weights.
    model.backpropagation(Y=model.Y_train)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {cost.item():.4f}")
