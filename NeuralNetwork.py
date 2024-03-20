import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pre_process_data():
    # Define the path to your Excel file
    excel_file = 'Dry_Bean_Dataset.xlsx'

    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)

    y = df['Class'].values
    
    # Drop the 'Class' column from the DataFrame
    df = df.drop(columns=['Class'])

    # Convert the DataFrame into a numpy array
    data = df.to_numpy()

    # Normalize the data
    data = data / np.max(data, axis=0)

    return data, y

def split_data(data, y, test_size=0.2, random_state=42):
    # Split the data and target into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

data, y = pre_process_data()
X_train, X_test, y_train, y_test = split_data(data, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train)

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function = activation

    def initialize_weights_and_bias(self):
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.bias1 = np.random.rand(1, self.hidden_size)
        self.bias2 = np.random.rand(1, self.output_size)
        
        assert self.W1.shape == (self.input_size, self.hidden_size)
        assert self.W2.shape == (self.hidden_size, self.output_size)
        assert self.bias1.shape == (1, self.hidden_size)
        assert self.bias2.shape == (1, self.output_size)
        
        return self.W1, self.W2, self.bias1, self.bias2
    
    
    
    