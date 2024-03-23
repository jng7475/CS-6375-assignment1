import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, activation='sigmoid', learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        prev_size = self.input_size
        for layer_size in self.hidden_layers:
            weight = np.random.randn(prev_size, layer_size)
            bias = np.zeros(layer_size)
            self.weights.append(weight)
            self.biases.append(bias)
            prev_size = layer_size
        self.weights.append(np.random.randn(prev_size, 1))
        self.biases.append(np.zeros(1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                activations.append(z)  # Output layer
            else:
                if self.activation == 'sigmoid':
                    activations.append(self.sigmoid(z))
                elif self.activation == 'tanh':
                    activations.append(self.tanh(z))
                elif self.activation == 'relu':
                    activations.append(self.relu(z))
        return activations

    def backward_propagation(self, X, y, activations):
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]

        delta = activations[-1] - y
        weight_gradients[-1] = np.dot(activations[-2].T, delta)
        bias_gradients[-1] = delta.sum(axis=0)

        for i in range(len(self.weights) - 2, -1, -1):
            if self.activation == 'sigmoid':
                delta = np.dot(delta, self.weights[i + 1].T) * activations[i + 1] * (1 - activations[i + 1])
            elif self.activation == 'tanh':
                delta = np.dot(delta, self.weights[i + 1].T) * (1 - np.square(activations[i + 1]))
            elif self.activation == 'relu':
                delta = np.dot(delta, self.weights[i + 1].T) * (activations[i + 1] > 0)

            weight_gradients[i] = np.dot(activations[i].T, delta)
            bias_gradients[i] = delta.sum(axis=0)

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def train(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=10):
        best_val_acc = 0.0
        patience_counter = 0
        for epoch in range(self.epochs):
            activations = self.forward_propagation(X)
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations)
            self.update_weights(weight_gradients, bias_gradients)

            # Calculate training accuracy
            y_train_pred = self.predict(X)
            train_accuracy = np.mean(y_train_pred.ravel() == y)

            # Calculate validation accuracy (if validation data is provided)
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_accuracy = np.mean(y_val_pred.ravel() == y_val)

                # Early stopping
                if early_stopping:
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Print progress
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Accuracy: {train_accuracy * 100:.2f}%", end="")
            if X_val is not None and y_val is not None:
                print(f", Validation Accuracy: {val_accuracy * 100:.2f}%", end="")
            print()

    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[-1] >= 0.5

def preprocess_data():
    # Load the Excel file into a pandas DataFrame
    excel_file = 'Dry_Bean_Dataset.xlsx'
    df = pd.read_excel(excel_file)

    # Encode the target variable (Class)
    class_mapping = {label: index for index, label in enumerate(df['Class'].unique())}
    df['Class'] = df['Class'].map(class_mapping)

    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Normalize the features
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / (test_size + val_size), random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Preprocess the data
    X, y = preprocess_data()

    # Split the data into training, validation, and testing sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")

    # Create a neural network with one hidden layer of 10 neurons
    nn = NeuralNetwork(input_size=X_train.shape[1], hidden_layers=[10], activation='sigmoid')

    # Train the neural network with early stopping
    nn.train(X_train, y_train.reshape(-1, 1), X_val, y_val.reshape(-1, 1), early_stopping=True, patience=10)

    # Evaluate the neural network on the testing set
    y_test_pred = nn.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test)
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()