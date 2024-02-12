import numpy as np
import random
import pickle

class SimpleNeuralNetwork:
    def __init__(self, neural_networks = []):
        if not(neural_networks):
            # Adjusted scale for weights and biases initialization
            self.weights_1      = np.random.uniform(-1, 1, size=(11, 10))
            self.biases_1       = np.random.uniform(-1,1, size= (1, 10))
            self.weights_2      =  np.random.uniform(-1, 1, size=(10, 10))
            self.biases_2       = np.random.uniform(-1,1, size= (1, 10))
            self.weights_output = np.random.uniform(-1,1, size= (10, 1))
            self.biases_output  = np.random.uniform(-1,1, size= (1, 1))
        else:
            if len(neural_networks) == 5:
                probabilities = [0.4, 0.2, 0.2, 0.1, 0.1]
            else:
                probabilities = [1]

            elements = list(range(0, len(neural_networks)))
            index = random.choices(elements, probabilities)[0]
            #index = random.randint(1, len(neural_networks))
            nn = neural_networks[index]
            self.weights_1 = nn.weights_1 + np.random.randn(11, 10) * 0.1
            self.biases_1 = nn.biases_1 + np.random.randn(10) * 0.1
            self.weights_2 = nn.weights_2 + np.random.randn(10, 10) * 0.1
            self.biases_2 = nn.biases_2 + np.random.randn(10) * 0.1
            self.weights_output = nn.weights_output + np.random.randn(10, 1) * 0.1
            self.biases_output = nn.biases_output + np.random.randn(1) * 0.1

    def save_to_memory(self, file_path=r'C:\Users\Vered\PycharmProjects\pythonProject1\SimpleNeuralNetwork.pkl'):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Saved instance to {file_path}")

    def load_from_memory(self, file_path=r'C:\Users\Vered\PycharmProjects\pythonProject1\SimpleNeuralNetwork.pkl'):
        with open(file_path, 'rb') as file:
            instance = pickle.load(file)
        print(f"Loaded instance from {file_path}")
        self.__dict__.update(instance.__dict__)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Optionally, define tanh or ReLU activation function
    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def forward_pass(self, input_layer):
        # Using tanh or ReLU for the hidden layers
        layer_1_output = self.tanh(np.dot(input_layer, self.weights_1) + self.biases_1)
        layer_2_output = self.tanh(np.dot(layer_1_output, self.weights_2) + self.biases_2)
        # Sigmoid for the output layer
        output = self.sigmoid(np.dot(layer_2_output, self.weights_output) + self.biases_output)
        return output

def nn(model, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11):
    # Prepare the input layer from the given inputs
    input_layer = np.array([in1, in2, in3, in4, in5, in6, in7, in8,in9, in10, in11])

    # Perform the forward pass through the model and return the output
    return model.forward_pass(input_layer)