Name: Thien Nguyen
NetID: DXN210021

Name: Trung "Jason" Nguyen
NetID: ttn190009

Assignment 1
CS6375 - Machine Learnning

Link to dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

This package including:
	README.TXT
	NeuralNetwork.py


How to run:
	1. Download the dataset from the link above
	2. Extract the dataset to the same folder as the NeuralNetwork.py
	3. Open the terminal and navigate to the folder containing the NeuralNetwork.py
	4. Run the following command:
		python3 NeuralNetwork.py
	
In the main code, we set test_size=0.25, which means 25% of the data will be used for testing, and the remaining 75% will be used for training.

The reasoning behind this choice is as follows:

Training Set Size: A larger training set (75% of the data) allows the model to learn more complex patterns and relationships in the data, which is particularly important for tasks with high-dimensional or complex data.

Testing Set Size: A test set size of 25% is considered large enough to provide a reliable estimate of the model's performance on unseen data, while not being too small to lead to high variance in the performance estimates.

Balance: The 75/25 split strikes a good balance between having a large training set for effective learning and a reasonably sized testing set for reliable performance evaluation.


