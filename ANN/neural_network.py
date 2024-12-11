import numpy as np


# we will use a class so that we can create multiple instances of a neural network
# this class contains the characteristics of a neural network

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, learning_rate=0.01, bias=0.5, threshold=0.01):
        """
        Input:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            hidden_layers (list of int): List specifying the number of neurons in each hidden layer.
            learning_rate (float): Learning rate for weight adjustments.
            bias (float): Bias added to each layer
            threshold (float): Threshold for error to stop training
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.bias = bias
        self.threshold = threshold
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        # initialize the weights
        weights = []
        layer_sizes = [self.input_size]+ self.hidden_layers + [self.output_size] 

        for i in range (len(layer_sizes) - 1):
            weight_matrix = np.random.uniform(-1.0, 1.0, (layer_sizes[i], layer_sizes[i + 1])) * 0.01
            weights.append(weight_matrix)

        return weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        """
        input:
            inputs (array): data (1D array of size input_size)

        output:
            activations (list): activation list for each of the layers
        """

        activations = [inputs]
        # step 1 - calculate the weighted sum
        for i in range(len(self.weights)):
            weighted_sum = np.dot(activations[-1], self.weights[i]) + self.bias

            # step 2 - apply activation function (sigmoid)
            activation = self.sigmoid(weighted_sum)
            activations.append(activation)
        return activations

    def calculate_error(self, predicted, actual):
        # step 3 - calculate the error using mean squared error
        return np.mean((predicted - actual) ** 2)
    
    def back_propagation(self, activations, actual):

        # step 4 - calculate error of output layer. activation of the previous 
        error = activations[-1] - actual
        delta = error * self.sigmoid_derivative(activations[-1])  # Delta for the output layer
        deltas = [delta]  # List of deltas for each layer
        
        # Backpropagate error to hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas.insert(0, delta)  # Insert at the beginning for backward order
        
        # Update weights based on deltas
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].reshape(-1, 1), deltas[i].reshape(1, -1))

    def train(self, data, targets, max_cycles=1000):
        for cycle in range(max_cycles):
            total_error = 0
            for i in range(len(data)):
                # step 1 - Forward propagation (weighted sum and applying sigmoid)
                activations = self.forward_propagation(data[i])
                
                # step 1 - Calculate error using MSE
                error = self.calculate_error(activations[-1], targets[i])
                total_error += error
                
                # Backpropagation and weight update - Calculate the output error and backpropagate so it can adjust the weights
                # then each weight will be adjusted by minusing (learning rate * the gradient in the deltas)
                self.back_propagation(activations, targets[i])

                # Log input, output, error, and weight changes
                print(f"Input: {data[i]}")
                print(f"Predicted Output: {activations[-1]}")
                print(f"Actual Output: {targets[i]}")
                print(f"Error: {error}")
                print(f"Weights after update: {self.weights}")
                print("-" * 30)

            # Stop training if total error is below the threshold
            if total_error < self.threshold:
                print(f"Training complete after {cycle + 1} cycles with error {total_error}")
                break
        pass

# Generate synthetic training data with a pattern
np.random.seed(42)  # For reproducibility

# Generate 100 input samples with 2 features each. example: temperature and humidity
data = np.random.rand(100, 2)

# this will hold the expected outputs for each sample. sum of inputs plus some noise, to give the network something to learn
targets = []
for x in data:
    target_value = sum(x) + 0.1 * np.random.randn()  # Sum the two input features and add noise, scale it down -0.1 to 0.1
    targets.append([target_value])  # Append as a single-element list

# Convert to a NumPy array after the loop
targets = np.array(targets)
# Define the neural network
input_size = 2        # Number of input features
output_size = 1       # Number of output neurons
hidden_layers = [3, 3]  # Two hidden layers with 3 neurons each
learning_rate = 0.1   # Learning rate for weight updates
bias = 0.5            # Bias added to each layer
threshold = 0.01      # Error threshold for stopping training

# Initialize the network
nn = NeuralNetwork(input_size, output_size, hidden_layers, learning_rate, bias, threshold)

# Train the network on the generated data
nn.train(data, targets, max_cycles=1000)
