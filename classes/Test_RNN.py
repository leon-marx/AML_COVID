# This file tests, wether the RNN and LSTM classes work as expected using data generated by the Sampler class.

# Package Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# Own Imports
import Datahandler
import LSTM_Model
import RNN_Model

# Parameters
input_size = 1
hidden_size = 256
output_size = 1
num_layers = 2
nonlinearity = "tanh"
dropout = 0.5
lower_lims = {
    "D": 1,
    "r": 0.0,
    "d": 7,
    "N_init": 0,
    "epsilon": 0.0
}
upper_lims = {
    "D": 5,
    "r": 0.5,
    "d": 21,
    "N_init": 20,
    "epsilon": 0.5
}
N = 1000  # population size
T = 50  # length of the simulation
version = "V3"
device = "cuda" if torch.cuda.is_available() else "cpu"
K = 10  # number of simulations sampled from the PP ranges
L = 20  # length of the slices
B = 10  # number of slices per simulation
n_epochs = 10000
learning_rate = 0.0001
test_batch_size = 4

# Models
print("Initializing Models")
myrnn = RNN_Model.RNN(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size,
                      num_layers=num_layers,
                      nonlinearity=nonlinearity)

mylstm = LSTM_Model.LSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         output_size=output_size,
                         num_layers=num_layers,
                         dropout=dropout)

mysampler = Datahandler.Sampler(lower_lims=lower_lims,
                                upper_lims=upper_lims,
                                N=N,
                                T=T,
                                version=version,
                                device=device)

# Data Generation
print("Generating Data")
batch, pandemic_parameters, starting_points = mysampler(K, L, B)
training_data = batch[:,:-test_batch_size,...]
test_data = batch[:,-test_batch_size:,...]
training_PP = pandemic_parameters[:,:-test_batch_size,...]
test_PP = pandemic_parameters[:,-test_batch_size:,...]

# RNN Training
print("Training RNN")
myrnn.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=myrnn.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    myrnn.train_model(training_data=training_data,training_PP=training_PP, loss_fn=loss_fn, optimizer=optimizer)
    if epoch % 100 == 0:
        myrnn.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)

# Plotting RNN Predictions
print("Plotting RNN Predictions")
x = np.arange(L-1)
plt.figure(figsize=(12, 8))
for i in range(test_batch_size):
    plt.subplot(2, 2, i+1)
    test_slice_X = test_data[:,i,...].view(L, 1, -1)[:L-1]
    test_slice_y = test_data[:,i,...].view(L, 1, -1)[L-1].to("cpu").view(-1).detach().numpy()
    PP_test_slice = test_PP[:,i,...].view(L, 1, 5)[:L-1].to(device)
    pred = myrnn.predict(test_slice_X, PP_test_slice).to("cpu").view(-1).detach().numpy()
    plt.plot(x, test_slice_X.to("cpu").view(-1), color="C0", label="Test Set")
    plt.scatter(L, pred, color="C1", label="Prediction")
    plt.scatter(L, test_slice_y, color="C0", marker="x", label="Truth")
    plt.legend()
plt.savefig("RNN_Predictions.png")
# plt.show()

# LSTM Training
print("Training LSTM")
mylstm.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=mylstm.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    mylstm.train_model(training_data=training_data,training_PP=training_PP, loss_fn=loss_fn, optimizer=optimizer)
    if epoch % 100 == 0:
        mylstm.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)

# Plotting LSTM Predictions
print("Plotting LSTM Predictions")
x = np.arange(L-1)
plt.figure(figsize=(12, 8))
for i in range(test_batch_size):
    plt.subplot(2, 2, i+1)
    test_slice_X = test_data[:,i,...].view(L, 1, -1)[:L-1]
    test_slice_y = test_data[:,i,...].view(L, 1, -1)[L-1].to("cpu").view(-1).detach().numpy()
    PP_test_slice = test_PP[:,i,...].view(L, 1, 5)[:L-1].to(device)
    pred = mylstm.predict(test_slice_X, PP_test_slice).to("cpu").view(-1).detach().numpy()
    plt.plot(x, test_slice_X.to("cpu").view(-1), color="C0", label="Test Set")
    plt.scatter(L, pred, color="C1", label="Prediction")
    plt.scatter(L, test_slice_y, color="C0", marker="x", label="Truth")
    plt.legend()
plt.savefig("LSTM_Predictions.png")
# plt.show()