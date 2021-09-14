# This file test, wether the RNN and LSTM classes work as expected.

# Package Imports
import torch

# Own Imports
import RNN_Model
import LSTM_Model
import Datahandler

# Parameters
input_size = 1
hidden_size = 256
output_size = 1
num_layers = 2
nonlinearity = "tanh"
dropout = 0.5
mode = "Simulation"
params = {
    "D": 8,
    "N": 1000,
    "r": 0.1,
    "d": 14,
    "N_init": 5,
    "T": 30,
    "epsilon": 0.1,
    "version": "V2"
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Models
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

mydh = Datahandler.DataHandler(mode=mode,
                               params=params,
                               device=device)

print("Done!")
