# Jason Nguyen - ttn190009
# Thien Nguyen - DXN210021
# Link to dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNetwork():
    def __init__(self, input_layer, hidden_layer, output_layer, activation='sigmoid', beta=0.9):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            self.activation = self.tanh
        # Initialize weights and biases at each layer
        # Initialize weights using Xavier initialization
        xavier_stddev_input = np.sqrt(2.0 / (input_layer + hidden_layer))
        xavier_stddev_hidden = np.sqrt(2.0 / (hidden_layer + output_layer))

        self.parameters = {
            "W1": np.random.normal(0, xavier_stddev_input, (hidden_layer, input_layer)),
            "bias_at_hidden_layer": np.zeros((hidden_layer, 1)),
            "W2": np.random.normal(0, xavier_stddev_hidden, (output_layer, hidden_layer)),
            "bias_at_output_layer": np.zeros((output_layer, 1))
        }
        # cache to save values during forward pass to use in backward pass
        self.cache = {}

        # Initialize momentum terms for each parameter
        self.momentum = {
            "W1": np.zeros((hidden_layer, input_layer)),
            "bias_at_hidden_layer": np.zeros((hidden_layer, 1)),
            "W2": np.zeros((output_layer, hidden_layer)),
            "bias_at_output_layer": np.zeros((output_layer, 1))
        }
        self.beta = beta  # Momentum hyperparameter

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)

    def forward_pass(self, X):

        self.cache["input_neurons"] = X

        # net = wX + b
        self.cache["net_from_inputs"] = np.matmul(
            self.parameters["W1"], self.cache["input_neurons"].T) + self.parameters["bias_at_hidden_layer"]
        # output = activision(wX + b)
        self.cache["hidden_layer_neurons"] = self.activation(
            self.cache["net_from_inputs"])
        # net = wX + b
        self.cache["net_from_hidden_layer"] = np.matmul(
            self.parameters["W2"], self.cache["hidden_layer_neurons"]) + self.parameters["bias_at_output_layer"]
        # output = activision(wX + b)
        self.cache["output_layer_neurons"] = self.activation(
            self.cache["net_from_hidden_layer"])
        return self.cache["output_layer_neurons"]

    def backward_pass(self, y_original, learning_rate=0.01):
        # δ(output) = (t - o) * derivative(activation)
        delta_output = (y_original.T - self.cache["output_layer_neurons"]) * self.activation(
            self.cache["net_from_hidden_layer"], derivative=True)

        # ∆(W2) = learning_rate * (delta(output) * hidden_layer_neurons)
        delta_W2 = learning_rate * \
            np.dot(delta_output, self.cache["hidden_layer_neurons"].T)
        delta_bias2 = learning_rate * \
            np.sum(delta_output, axis=1, keepdims=True)

        # δ(hidden) = (W2.T * delta(output)) * derivative(activation)
        delta_hidden = np.dot(self.parameters["W2"].T, delta_output) * self.activation(
            self.cache["net_from_inputs"], derivative=True)

        # ∆W1 = learning_rate * (delta(hidden) * input_neurons)
        delta_W1 = learning_rate * \
            np.dot(delta_hidden, self.cache["input_neurons"])
        delta_bias1 = learning_rate * \
            np.sum(delta_hidden, axis=1, keepdims=True)

        # Update parameters using gradients and learning rate
        self.parameters["W2"] += delta_W2
        self.parameters["bias_at_output_layer"] += delta_bias2
        self.parameters["W1"] += delta_W1
        self.parameters["bias_at_hidden_layer"] += delta_bias1

        self.momentum_optimizer(delta_W2, delta_bias2, delta_W1, delta_bias1)

        # Update parameters using gradients and momentum
        self.parameters["W2"] += self.momentum["W2"]
        self.parameters["bias_at_output_layer"] += self.momentum["bias_at_output_layer"]
        self.parameters["W1"] += self.momentum["W1"]
        self.parameters["bias_at_hidden_layer"] += self.momentum["bias_at_hidden_layer"]

    def momentum_optimizer(self, delta_W2, delta_bias2, delta_W1, delta_bias1):
        # Update momentum
        self.momentum["W2"] = self.beta * \
            self.momentum["W2"] + (1 - self.beta) * delta_W2
        self.momentum["bias_at_output_layer"] = self.beta * \
            self.momentum["bias_at_output_layer"] + \
            (1 - self.beta) * delta_bias2
        self.momentum["W1"] = self.beta * \
            self.momentum["W1"] + (1 - self.beta) * delta_W1
        self.momentum["bias_at_hidden_layer"] = self.beta * \
            self.momentum["bias_at_hidden_layer"] + \
            (1 - self.beta) * delta_bias1

    def get_accuracy(self, y, output):
        y_predicted = np.argmax(output.T, axis=-1)
        y_original = np.argmax(y, axis=-1)
        return np.mean(y_predicted == y_original)

    def train(self, x_train, y_train, x_test, y_test, epochs=10,
              batch_size=64, l_rate=0.1):
        # number of trainings
        self.epochs = epochs
        # divide the data into batches
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)

        # Train
        for i in range(self.epochs):
            # Shuffle
            shuffle = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[shuffle]
            y_train_shuffled = y_train[shuffle]

            for j in range(num_batches):
                # begin index of the batch
                begin = j * self.batch_size
                # end index of the batch
                end = min(begin + self.batch_size, x_train.shape[0]-1)

                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                # Backpropagation algorithm
                # Forward pass
                output = self.forward_pass(x)
                # Backward pass
                self.backward_pass(y, l_rate)

            # Train data
            output = self.forward_pass(x_train)
            train_accuracy = self.get_accuracy(y_train, output)
            # Test data
            output = self.forward_pass(x_test)
            test_acc = self.get_accuracy(y_test, output)

            result = "Epoch {}: train acc = {:.2f}, test acc = {:.2f},"
            print(result.format(i+1, train_accuracy, test_acc))


def preprocess_data():
    # Load the Excel file into a pandas DataFrame
    excel_file = 'Dry_Bean_Dataset.xlsx'
    df = pd.read_excel(excel_file)

    # Encode the target variable (Class)
    class_mapping = {label: index for index,
                     label in enumerate(df['Class'].unique())}
    df['Class'] = df['Class'].map(class_mapping)

    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # One-hot encode the target variable
    num_classes = len(class_mapping)
    y = np.eye(num_classes)[y]

    # Normalize the features
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def main():
    # Preprocess the data
    X, y = preprocess_data()

    # Split the data into training and testing sets
    test_size = 0.25  # Ideal ratio for training and testing
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    print(f"Training set: {y_train.shape}, Testing set: {y_test.shape}")

    # Get the number of classes from the target variable
    num_classes = y_train.shape[1]
    hidden_layer = 10
    dnn = NeuralNetwork(16, hidden_layer, num_classes,
                        activation='sigmoid', beta=0.9)
    dnn1 = NeuralNetwork(16, hidden_layer, num_classes,
                         activation='tanh', beta=0.9)
    dnn2 = NeuralNetwork(16, hidden_layer, num_classes,
                         activation='relu', beta=0.9)
    learning_rate = 0.01
    a_batch_size = 32
    print("sigmoid")
    dnn.train(X_train, y_train, X_test, y_test,
              epochs=10, batch_size=a_batch_size, l_rate=learning_rate)
    print("tanh")
    dnn1.train(X_train, y_train, X_test, y_test,
               epochs=10, batch_size=a_batch_size, l_rate=learning_rate)
    print("relu")
    dnn2.train(X_train, y_train, X_test, y_test, epochs=10,
               batch_size=a_batch_size, l_rate=learning_rate)


if __name__ == "__main__":
    main()
