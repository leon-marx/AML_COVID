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
        self.input_size = input_size + 5
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        self.get_h0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)

    def forward(self, sequence, h_0=None):
        """
        sequence: should be a time series of shape (L, N, input_PP_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_PP_size: timeseries + PPs, generally 1 + 5 = 6
        h_0: optional, can specify initial hidden layer values with shape (num_layers, N, hidden_size) with:
            num_layers: as above, generally 1
            N: batch_size, generally 1
            hidden_size: as above, e.g. 256
            if h_0 == None: initialize with zeros
        """
        N = sequence.shape[1]
        if h_0 == None:
            h_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(device)
        sequence = sequence.to(device)
        output, h_final = self.rnn(sequence, h_0)
        output = self.linear(output[-1]).view(1, N, self.output_size)
        return output
        
    def predict(self, sequence, PP_input):
        """
        sequence: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        PP_input: pandemic parameters, tensor of shape (L, N, 5) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        """
        PP_sequence = torch.cat((sequence, PP_input), dim=2)
        self.eval()
        with torch.no_grad():
            # Compute prediction error
            h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size)
            pred = self.forward(PP_sequence, h_0=h_0)
        return pred

    def train_model(self, training_data, training_PP, loss_fn, optimizer, verbose=False):
        """
        training_data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            input_size: as above, generally 1
        training_PP: pandemic parameters, tensor of shape (L, N, 5) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        loss_fn: use the loss functions provided by pytorch
        optimizer: use the optimizers provided by pytorch
        verbose: set True to print out the Training Loss
        """
        training_data = training_data.to(device)
        training_PP = training_PP.to(device)
        X_data = training_data[:-self.output_size]
        y_data = training_data[-self.output_size:]
        X_PP = training_PP[:-self.output_size]
        training_seq = torch.cat((X_data, X_PP), dim=2)
        
        self.train()

        # Compute prediction error
        h_0 = self.get_h0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
        pred = self.forward(training_seq, h_0=h_0)
        loss = loss_fn(pred, y_data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print("Training Loss:", loss.item())

    def test_model(self, test_data, test_PP, loss_fn):
        """
        test_data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            input_size: as above, generally 1
        test_PP: pandemic parameters, tensor of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        loss_fn: use the loss functions provided by pytorch
        """
        test_data = test_data.to(device)
        test_PP = test_PP.to(device)
        X_data = test_data[:-self.output_size]
        y_data = test_data[-self.output_size:]
        X_PP = test_PP[:-self.output_size]
        test_seq = torch.cat((X_data, X_PP), dim=2)

        test_loss = 0
        self.eval()
        with torch.no_grad():
            # Compute prediction error
            h_0 = self.get_h0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
            pred = self.forward(test_seq, h_0=h_0)
            test_loss = loss_fn(pred, y_data).item()
        print("Average Test Loss:", test_loss)
        return test_loss

# Example use

print("Running on:", device)

from Datahandler import DataHandler

import matplotlib.pyplot as plt
import numpy as np
B = 500  # batch size
L = 20  # sequence length
params = {"N": 100,
          "D": 5,
          "r": 0.2,
          "d": 14,
          "N_init": 1,
          "epsilon": 0.4,
          "version": "V2",
          "T": L + 1}
DH = DataHandler(mode="Simulation", params=params, device=device)
data, starting_points, PP_data  = DH(B,L)
print("Data:            ", data.shape)
training_data = data[:,:350,...]
test_data = data[:,350:,...]
print("Training Data:   ", training_data.shape)
print("Test Data:       ", test_data.shape)

print("PP Data:         ", PP_data.shape)
PP_training_data = PP_data[:,:350,...]
PP_test_data = PP_data[:,350:,...]
print("PP Training Data:", PP_training_data.shape)
print("PP Test Data:    ", PP_test_data.shape)

input_size = 1
hidden_size = 256
output_size = 1
num_layers = 2
nonlinearity = "tanh"

n_epochs = 1000
learning_rate = 0.0001


MyRNN = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, nonlinearity=nonlinearity).to(device)

# Print model and its parameters
"""
for name, param in MyRNN.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=MyRNN.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    MyRNN.train_model(training_data=training_data,training_PP=PP_training_data, loss_fn=loss_fn, optimizer=optimizer)
    if epoch % 100 == 0:
        MyRNN.test_model(test_data=test_data, test_PP=PP_test_data, loss_fn=loss_fn)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    test_slice_X = test_data[:,i,...].view(L, 1, -1)[:L-1]
    test_slice_y = test_data[:,i,...].view(L, 1, -1)[L-1].to("cpu").view(-1).detach().numpy()
    PP_test_slice = PP_test_data[:,i,...].view(L, 1, 5)[:L-1].to(device)
    pred = MyRNN.predict(test_slice_X, PP_test_slice).to("cpu").view(-1).detach().numpy()
    plt.plot(np.arange(L-1), test_slice_X.to("cpu").view(-1), color="C0", label="Test Set")
    plt.scatter(L, pred, color="C1", label="Prediction")
    plt.scatter(L, test_slice_y, color="C0", marker="x", label="Truth")
    plt.legend()
plt.show()
