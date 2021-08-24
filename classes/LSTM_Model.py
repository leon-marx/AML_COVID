import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        input_size: number of features in input, generally 1
        hidden_size: number of hidden neurons, e.g. 256
        output_size: how many days to predict, generally 1
        num_layers: number of stacked LSTMs, generally 1
        dropout: dropout-probability, 0 corresponds to no dropout
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.get_h0 = nn.Linear(in_features=5, out_features=num_layers * hidden_size)
        self.get_c0 = nn.Linear(in_features=5, out_features=num_layers * hidden_size)

    def forward(self, sequence, inits=None):
        """
        sequence: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        inits: optional, can specify initial hidden and cell layer values with shape (2, num_layers, N, hidden_size) with:
            2: corresponds to (h0, c0)
            num_layers: as above, generally 1
            N: batch_size, generally 1
            hidden_size: as above, e.g. 256
            if h_0 == None: initialize with zeros
        """
        N = sequence.shape[1]
        if inits == None:
            h_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(device)
            c_0 = torch.zeros((self.num_layers, N, self.hidden_size)).to(device)
            inits = (h_0, c_0)
        sequence = sequence.to(device)
        output, (h_final, c_final) = self.lstm(sequence, inits)
        output = self.linear(output[-1]).view(1, N, self.input_size)
        return output

    def train_model(self, training_data, training_PP, loss_fn, optimizer, verbose=False):
        """
        training_data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size
            input_size: as above, generally 1
        training_PP: pandemic parameters, tensor of shape (N, 5) with:
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
        self.train()

        # Compute prediction error
        h_0 = self.get_h0(training_PP).view(self.num_layers, training_PP.shape[0], self.hidden_size)
        c_0 = self.get_c0(training_PP).view(self.num_layers, training_PP.shape[0], self.hidden_size)
        pred = self.forward(X_data, inits=(h_0, c_0))
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
        test_PP: pandemic parameters, tensor of shape (N, 5) with:
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
        test_loss = 0
        self.eval()
        with torch.no_grad():
            # Compute prediction error
            h_0 = self.get_h0(test_PP).view(self.num_layers, test_PP.shape[0], self.hidden_size)
            c_0 = self.get_c0(test_PP).view(self.num_layers, test_PP.shape[0], self.hidden_size)
            pred = self.forward(X_data, inits=(h_0, c_0))
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
PP_test_data = PP_data[350:,...]
PP_training_data = PP_data[:350,...]
print("PP Training Data:", PP_training_data.shape)
print("PP Test Data:    ", PP_test_data.shape)

input_size = 1
hidden_size = 256
output_size = 1
num_layers = 2
dropout = 0.1

n_epochs = 1000
learning_rate = 0.00003


MyLSTM = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)

# Print model and its parameters

for name, param in MyLSTM.named_parameters():
    if param.requires_grad:
        print(name)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=MyLSTM.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    MyLSTM.train_model(training_data=training_data,training_PP=PP_training_data, loss_fn=loss_fn, optimizer=optimizer)
    if epoch % 100 == 0:
        MyLSTM.test_model(test_data=test_data, test_PP=PP_test_data, loss_fn=loss_fn)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    test_slice = test_data[:,i,...].view(L, 1, -1)
    MyLSTM.eval()
    pred = MyLSTM.forward(test_slice).to("cpu").view(-1).detach().numpy()
    plt.plot(np.arange(L), test_slice.to("cpu").view(-1), color="C0", label="Test Set")
    plt.scatter(L+1, pred, color="C1", label="Prediction")
    plt.legend()
plt.show()
