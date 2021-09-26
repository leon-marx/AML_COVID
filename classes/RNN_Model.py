import torch
from torch import nn
from Datahandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np
import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity, foretime=3, backtime=20):
        """
        input_size: number of features in input, generally 1
        hidden_size: number of hidden neurons, e.g. 256
        num_layers: number of stacked RNNs, generally 1
        nonlinearity: choose "tanh" or "relu"
        """
        super(RNN, self).__init__()
        self.input_size = input_size + 5
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.foretime = foretime
        self.backtime = backtime
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.get_h0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)

    def forward(self, sequence, h_0=None, full=False):
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
        output = self.linear(output[-1]).view(1, N, 1)
        if full:
            return output, h_final
        return output
        
    def forward_long(self, sequence, PP_input, n_days):
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
        n_days: number of days to predict in the future
        """
        sequence = sequence.to(device)
        PP_input = PP_input.to(device)
        preds = torch.zeros(size=(n_days, sequence.shape[1], self.input_size-5))
        for i in range(n_days):
            PP_sequence = torch.cat((sequence, PP_input), dim=2) #19,1,6 (number of days, ..., 6 dimensions [5PP,1value]); 10,1,6
            if i == 0:
                h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                pred, h_final = self.forward(PP_sequence, h_0, full=True)
                new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                preds[i] = pred
            else:
                pred, h_final = self.forward(new_PP_seq, h_final, full=True)
                new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                preds[i] = pred
        return preds

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

    def predict_long(self, sequence, PP_input, n_days):
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
        n_days: number of days to predict in the future
        """
        sequence = sequence.to(device)
        PP_input = PP_input.to(device)
        preds = torch.zeros(size=(n_days, sequence.shape[1], self.input_size-5))
        for i in range(n_days):
            PP_sequence = torch.cat((sequence, PP_input), dim=2) #19,1,6 (number of days, ..., 6 dimensions [5PP,1value]); 10,1,6
            self.eval()
            with torch.no_grad():
                if i == 0:
                    h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                    pred, h_final = self.forward(PP_sequence, h_0=h_0, full=True)
                    new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                    preds[i] = pred
                else:
                    pred, h_final = self.forward(new_PP_seq, h_final, full=True)
                    new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                    preds[i] = pred
        return preds

    def train_model(self, training_X, training_PP, training_y, loss_fn, optimizer):
        """
        Implements one training step for a given time-series
        training_X: should be a time series of shape (backtime, N, input_size) with:
            backtime: as in init
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
        training_y: should be a time series of shape (foretime, N, input_size) with:
            foretime: as in init
            N: batch_size
            input_size: as above, generally 1
        loss_fn: use the loss functions provided by pytorch
        optimizer: use the optimizers provided by pytorch
        """
        # Put on GPU if possible
        training_X = training_X.to(device)
        training_PP = training_PP.to(device)
        training_y = training_y.to(device)
        
        # Predict value for next timestep and compute prediction error
        self.train()
        pred = self.forward_long(training_X, training_PP, self.foretime).to(device)
        loss = loss_fn(pred, training_y).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()   

    def test_model(self, test_X, test_PP, test_y, loss_fn):
        """
        test_X: should be a time series of shape (backtime, N, input_size) with:
            backtime: as in init
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
        test_y: should be a time series of shape (foretime, N, input_size) with:
            foretime: as in init
            N: batch_size
            input_size: as above, generally 1
        loss_fn: use the loss functions provided by pytorch
        """
        # Put on GPU if possible
        test_X = test_X.to(device)
        test_PP = test_PP.to(device)
        test_y = test_y.to(device)

        self.eval()
        with torch.no_grad():
            # Compute prediction error
            pred = self.predict_long(test_X, test_PP, self.foretime).to(device)
            test_loss = loss_fn(pred, test_y).item()
        return test_loss
    
    def load_model(self, path="Trained_RNN_Model"):
        return torch.load(path)


# Example use
if __name__ == "__main__":



    import Datahandler

    # Parameters
    input_size = 1
    hidden_size = 256
    output_size = 1
    num_layers = 2
    nonlinearity = "tanh"
    lower_lims = {
        "D": 1,
        "r": 0.0,
        "d": 3,
        "N_init": 1,
        "epsilon": 0.0
    }
    upper_lims = {
        "D": 10,
        "r": 0.5,
        "d": 21,
        "N_init": 5,
        "epsilon": 0.7
    }
    N = 1000  # population size
    T = 50  # length of the simulation   AS BIG AS POSSIBLE WITHOUT FLAT
    version = "V3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = 50  # number of simulations sampled from the PP ranges   AS BIG AS POSSIBLE
    L = T - 5  # length of the slices
    B = 5  # number of slices per simulation
    n_epochs = 10000
    learning_rate = 0.0001
    test_batch_size = 4
    backtime = 20  # number of days the network gets to see before prediction
    foretime = 3  # number of days to predict for long predictions

    # Models
    print("Initializing Models")
    myrnn = RNN(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity=nonlinearity,
                            foretime=foretime,
                            backtime=backtime)

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
        print(f"Starting epoch {epoch}")
        myrnn.train_model(training_data=training_data, training_PP=training_PP, loss_fn=loss_fn, optimizer=optimizer)
        if epoch % 100 == 0:
            myrnn.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)

    # Save Model
    torch.save(myrnn.state_dict(), "Trained_RNN_Model")
