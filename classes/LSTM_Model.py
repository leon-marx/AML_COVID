import torch
from torch import nn
from Datahandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, foretime=3, backtime=20):
        """
        input_size: number of features in input, generally 1
        hidden_size: number of hidden neurons, e.g. 256
        num_layers: number of stacked LSTMs, generally 1
        dropout: dropout-probability, 0 corresponds to no dropout
        """
        super(LSTM, self).__init__()
        self.input_size = input_size + 5
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.foretime = foretime
        self.backtime = backtime
        self.lstm = nn.LSTM(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.get_h0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)
        self.get_c0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)

    def forward(self, sequence, inits=None, full=False):
        """
        sequence: should be a time series of shape (L, N, input_PP_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_PP_size: timeseries + PPs, generally 1 + 5 = 6
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
        output = self.linear(output[-1]).view(1, N, 1)
        if full:
            return output, (h_final, c_final)
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
                c_0 = self.get_c0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                pred, (h_final, c_final) = self.forward(PP_sequence, inits=(h_0, c_0), full=True)
                new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                preds[i] = pred
            else:
                pred, (h_final, c_final) = self.forward(new_PP_seq, inits=(h_final, c_final), full=True)
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
            c_0 = self.get_c0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size)
            pred = self.forward(PP_sequence, inits=(h_0, c_0))
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
                    c_0 = self.get_c0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                    pred, (h_final, c_final) = self.forward(PP_sequence, inits=(h_0, c_0), full=True)
                    new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                    preds[i] = pred
                else:
                    pred, (h_final, c_final) = self.forward(new_PP_seq, inits=(h_final, c_final), full=True)
                    new_PP_seq = torch.cat((pred, PP_input[-1].view(1, PP_input.shape[1], 5)), dim=2)
                    preds[i] = pred
        return preds

    def train_model_old(self, training_data, training_PP, loss_fn, optimizer, epoch=None, verbose=False):
        """
        Implements one training step for a given time-series
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
        # Put on GPU if possible
        training_data = training_data.to(device)
        training_PP = training_PP.to(device)

        # Split data 
        X_data = training_data[:-1]
        y_data = training_data[-1:]
        X_PP = training_PP[:-1]

        # Combine PP and timeseries with cumulative COVID-19 cases 
        training_seq = torch.cat((X_data, X_PP), dim=2)

        # Initialize states h(0) and c(0) with PPs, predict value for next timestep and compute prediction error
        self.train()
        h_0 = self.get_h0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
        c_0 = self.get_c0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
        pred = self.forward(training_seq, inits=(h_0, c_0))
        loss = loss_fn(pred, y_data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Epoch: {epoch} | Training Loss:", loss)

    def train_model(self, training_data, training_PP, loss_fn, optimizer, epoch=None, verbose=False):
        """
        Implements one training step for a given time-series
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
        L = training_data.shape[0]
        N = training_data.shape[1]

        # Put on GPU if possible
        training_data = training_data.to(device)
        training_PP = training_PP.to(device)
        
        # Split data into many slices
        X_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        X_PP_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), 5))
        y_data = torch.zeros(size=(self.foretime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        for i in range(L-(self.backtime+self.foretime)):
            X = training_data[i:i+self.backtime]
            X_PP = training_PP[i:i+self.backtime]
            y = training_data[i+self.backtime:i+self.backtime+self.foretime]
            X_data[:, i*N:(i+1)*N, :] = X
            X_PP_data[:, i*N:(i+1)*N, :] = X_PP
            y_data[:, i*N:(i+1)*N, :] = y

        # Predict value for next timestep and compute prediction error
        self.train()
        pred = self.forward_long(X_data, X_PP_data, self.foretime)
        loss = loss_fn(pred, y_data).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            # print(f"Epoch: {epoch} | Training Loss:", loss)
            return loss.item()

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
        L = test_data.shape[0]
        N = test_data.shape[1]

        # Put on GPU if possible
        test_data = test_data.to(device)
        test_PP = test_PP.to(device)
        
        # Split data into many slices
        X_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        X_PP_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), 5))
        y_data = torch.zeros(size=(self.foretime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        for i in range(L-(self.backtime+self.foretime)):
            X = test_data[i:i+self.backtime]
            X_PP = test_PP[i:i+self.backtime]
            y = test_data[i+self.backtime:i+self.backtime+self.foretime]
            X_data[:, i*N:(i+1)*N, :] = X
            X_PP_data[:, i*N:(i+1)*N, :] = X_PP
            y_data[:, i*N:(i+1)*N, :] = y

        self.eval()
        with torch.no_grad():
            # Compute prediction error
            pred = self.predict_long(X_data, X_PP_data, self.foretime)
            test_loss = loss_fn(pred, y_data).item()
        print("Average Test Loss:", test_loss)

        return test_loss
    
    def load_model(self, path="Trained_LSTM_Model"):
        return torch.load(path)

    
# Example use
if __name__ == "__main__":
    


    import Datahandler

    # Parameters
    input_size = 1
    hidden_size = 256
    output_size = 1
    num_layers = 2
    dropout = 0.5
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
    mylstm = LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            foretime=foretime,
                            backtime=backtime)

    mysampler = Datahandler.Sampler(lower_lims=lower_lims,
                                    upper_lims=upper_lims,
                                    N=N,
                                    T=T,
                                    version=version,
                                    device=device)

    mylstm.load_state_dict(mylstm.load_model())
    


    # Data Generation
    print("Generating Data")
    batch, pandemic_parameters, starting_points = mysampler(K, L, B)
    training_data = batch[:,:-test_batch_size,...]
    test_data = batch[:,-test_batch_size:,...]
    training_PP = pandemic_parameters[:,:-test_batch_size,...]
    test_PP = pandemic_parameters[:,-test_batch_size:,...]
    
    loss_fn = torch.nn.MSELoss()
    mylstm.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)

'''

    # LSTM Training
    print("Training LSTM")
    mylstm.to(device)
    
    optimizer = torch.optim.Adam(params=mylstm.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}")
        mylstm.train_model(training_data=training_data, training_PP=training_PP, loss_fn=loss_fn, optimizer=optimizer)
        if epoch % 100 == 0:
            mylstm.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)

    # Save Model
    torch.save(mylstm.state_dict(), "Trained_LSTM_Model")
    '''


