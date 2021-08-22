import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nonlinearity):
        """
        input_size: number of features in input, generally 1
        hidden_size: number of hidden neurons, e.g. 256
        output_size: how many days to predict, generally 1
        num_layers: number of stacked RNNs, generally 1
        nonlinearity: choose "tanh" or "relu"
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, sequence, h_0=None):
        """
        sequence: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        h_0: optional, can specify initial hidden layer values with shape (num_layers, N, hidden_size) with:
            num_layers: as above, generally 1
            N: batch_size, generally 1
            hidden_size: as above, e.g. 256
            if h_0 == None: initialize with zeros
        """
        if h_0 == None:
            N = sequence.shape[1]
            h_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(device)
        sequence = sequence.to(device)
        output, h_final = self.rnn(sequence, h_0)
        output = self.linear(output[-1]).view(1, N, self.input_size)
        return output

    def train_model(self, training_data, loss_fn, optimizer, h_0=None, verbose=False):
        """
        training_data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        loss_fn: use the loss functions provided by pytorch
        optimizer: use the optimizers provided by pytorch
        h_0: optional, can specify initial hidden layer values with shape (num_layers, N, hidden_size) with:
            num_layers: as above, generally 1
            N: batch_size, generally 1
            hidden_size: as above, e.g. 256
            if h_0 == None: initialize with zeros
        verbose: set True to print out the Training Loss
        """
        training_data = training_data.to(device)
        X_data = training_data[:-self.output_size]
        y_data = training_data[-self.output_size:]
        self.train()

        # Compute prediction error
        pred = self.forward(X_data)
        loss = loss_fn(pred, y_data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print("Training Loss:", loss.item())

    def test_model(self, test_data, loss_fn, h_0=None):
        """
        test_data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        loss_fn: use the loss functions provided by pytorch
        h_0: optional, can specify initial hidden layer values with shape (num_layers, N, hidden_size) with:
            num_layers: as above, generally 1
            N: batch_size, generally 1
            hidden_size: as above, e.g. 256
            if h_0 == None: initialize with zeros
        """
        test_data = test_data.to(device)
        X_data = test_data[:-self.output_size]
        y_data = test_data[-self.output_size:]
        test_loss = 0
        self.eval()
        with torch.no_grad():
            # Compute prediction error
            pred = self.forward(X_data)
            test_loss += loss_fn(pred, y_data).item()
        test_loss /= test_data.shape[1]
        print("Average Test Loss:", test_loss)
        return test_loss

# Example use
"""
print("Running on:", device)

from Datahandler import DataHandler

### Path problem, easy fix (only for me)
import os
# print(os.getcwd())
os.chdir("./AML_COVID/classes")
###

import matplotlib.pyplot as plt
import numpy as np
DH = DataHandler(mode="Real", params={"file": "Germany.txt"}, device="cpu")
N = 500  # batch size
L = 100  # sequence length
data, starting_points  = DH(N,L)
print("Data:         ", data.shape)
training_data = data[:,:350,...]
test_data = data[:,350:,...]

print("Training Data:", training_data.shape)
print("Test Data:    ", test_data.shape)

input_size = 1
hidden_size = 256
output_size = 1
num_layers = 2
nonlinearity = "tanh"

n_epochs = 1000
learning_rate = 0.0001


MyRNN = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, nonlinearity=nonlinearity).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=MyRNN.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    MyRNN.train_model(training_data=training_data, loss_fn=loss_fn, optimizer=optimizer)
    if epoch % 100 == 0:
        MyRNN.test_model(test_data=test_data, loss_fn=loss_fn)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    test_slice = test_data[:,i,...].view(L, 1, -1)
    preds = []
    outliers = []
    pred_inds = []
    outlier_inds = []
    pred = MyRNN.forward(test_slice).to("cpu").view(-1).detach().numpy()
    plt.plot(np.arange(L), test_slice.view(-1), color="C0", label="Test Set")
    plt.scatter(L+1, pred, color="C1", label="Prediction")
    plt.legend()
plt.show()
"""
