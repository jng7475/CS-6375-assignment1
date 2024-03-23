import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNetwork():
    def __init__(self, input_layer, hidden_layer, output_layer, activation='sigmoid'):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            self.activation = self.tanh
        # Save all weights
        self.parameters = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "bias1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "bias2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        self.cache = {}

    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

    def forward_pass(self, x):
        # y = activision(wX + b)

        self.cache["input_neurons"] = x

        self.cache["net_from_inputs"] = np.matmul(
            self.parameters["W1"], self.cache["input_neurons"].T) + self.parameters["bias1"]
        self.cache["hidden_layer_neurons"] = self.activation(
            self.cache["net_from_inputs"])
        self.cache["net_from_hidden_layer"] = np.matmul(
            self.parameters["W2"], self.cache["hidden_layer_neurons"]) + self.parameters["bias2"]
        self.cache["output_layer_neurons"] = self.activation(
            self.cache["net_from_hidden_layer"])
        return self.cache["output_layer_neurons"]

    def backward_pass(self, y_true, learning_rate=0.01):
        # δ(output) = (y - y_true) * derivative(activation)
        delta_output = (y_true.T - self.cache["output_layer_neurons"]) * self.activation(
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
        self.parameters["bias2"] += delta_bias2
        self.parameters["W1"] += delta_W1
        self.parameters["bias1"] += delta_bias1

    def accuracy(self, y, output):
        y_pred = np.argmax(output.T, axis=-1)
        y_true = np.argmax(y, axis=-1)
        return np.mean(y_pred == y_true)

    def train(self, x_train, y_train, x_test, y_test, epochs=10,
              batch_size=64, optimizer='sgd', l_rate=0.1, beta=.9):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)

        # Initialize optimizer
        self.optimizer = optimizer

        start_time = time.time()
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                # Forward
                output = self.forward_pass(x)
                # Backward
                self.backward_pass(y, l_rate)

            # Train data
            output = self.forward_pass(x_train)
            train_accuracy = self.accuracy(y_train, output)
            # Test data
            output = self.forward_pass(x_test)
            test_acc = self.accuracy(y_test, output)
            # print epoch, train accuracy, test accuracy
            result = "Epoch {}: {:.2f}s, train acc={:.2f}, test acc={:.2f},"
            print(result.format(i+1, time.time()-start_time,
                                train_accuracy,
                                test_acc))


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
    dnn = NeuralNetwork(16, 10, num_classes, activation='sigmoid')
    dnn.train(X_train, y_train, X_test, y_test, batch_size=32,
              optimizer='sgd', l_rate=0.1, beta=.9)


if __name__ == "__main__":
    main()
