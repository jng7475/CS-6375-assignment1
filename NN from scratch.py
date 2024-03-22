import numpy as np
import matplotlib.pyplot as plt


class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_func='relu', learning_rate=0.01):
        np.random.seed(42)  # For reproducibility
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        self.activation_func = activation_func
        self.learning_rate = learning_rate
    
    def activation(self, x):
        if self.activation_func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_func == 'tanh':
            return np.tanh(x)
        elif self.activation_func == 'relu':
            return np.maximum(0, x)
    
    def activation_derivative(self, x):
        if self.activation_func == 'sigmoid':
            return x * (1 - x)
        elif self.activation_func == 'tanh':
            return 1 - np.power(x, 2)
        elif self.activation_func == 'relu':
            return np.where(x <= 0, 0, 1)

    def initialize(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return params
    
    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.weights1) + self.bias1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights2) + self.bias2
        self.A2 = self.activation(self.Z2)
        return self.A2

    def back_propagation(self, X, y, output):
        m = y.shape[0]
        dZ2 = output - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.weights2.T) * self.activation_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.weights1 -= self.learning_rate * dW1
        self.bias1 -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dW2
        self.bias2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward_propagation(X)
            self.back_propagation(X, y, output)

    def predict(self, X):
        output = self.forward_propagation(X)
        predictions = np.argmax(output, axis=1)
        return predictions

    def evaluate_accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
