import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import Dataset

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

    def predict_long(self, sequence, PP_input, n_days, long_PP=False):
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
        long_PP: the PP input is longer than the sequence, marking an expected change in policy
        """
        sequence = sequence.to(device)
        PP_input = PP_input.to(device)
        preds = torch.zeros(size=(n_days, sequence.shape[1], self.input_size-5))
        ###
        # THIS IS FOR TESTING, MIGHT BE BUGGY, BUT IF NOT ITS NICER
        long_PP = True
        ###
        if long_PP:
            for i in range(n_days):
                L_seq = sequence.shape[0]
                L_p = PP_input.shape[0] - L_seq
                PP_sequence = torch.cat((sequence, PP_input[:L_seq]), dim=2) #19,1,6 (number of days, ..., 6 dimensions [5PP,1value]); 10,1,6
                self.eval()
                with torch.no_grad():
                    PP_ind = min(L_seq+L_p-1, L_seq+i)
                    if i == 0:
                        h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                        c_0 = self.get_c0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                        pred, (h_final, c_final) = self.forward(PP_sequence, inits=(h_0, c_0), full=True)
                        new_PP_seq = torch.cat((pred, PP_input[PP_ind].view(1, PP_input.shape[1], 5)), dim=2)
                        preds[i] = pred
                    else:
                        pred, (h_final, c_final) = self.forward(new_PP_seq, inits=(h_final, c_final), full=True)
                        new_PP_seq = torch.cat((pred, PP_input[PP_ind].view(1, PP_input.shape[1], 5)), dim=2)
                        preds[i] = pred
        else:
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
    
    def load_model(self, path="Trained_LSTM_Model"):
        return self.state_dict = torch.load(path)

    def apply_prediction(self, real_data, real_PP, n_days):
        """
        real_Data: should be a time series of shape (L, N, input_size) with:
            L: sequence length, e.g. 50 days
            N: batch_size, generally 1
            input_size: as above, generally 1
        real_PP: pandemic parameters, tensor of shape (L+L_p, N, 5) with:
            L: sequence length, e.g. 50 days
            L_p: length one wants to predict. This is optional, if it is 0 the Net will assume constant PPs at the end
            N: batch_size, generally 1
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        n_days: how many days to predict
        """
        ###
        # IF THIS DOES NOT WORK; YOU HAVE TO CUT TO THE LAST self.backtime DAYS ONLY!!!!!!!!!!!!!!!
        # IF PREDICT LONG WORKS WITH long_PP=True ALWAYS; WE CAN SKIP THIS IF-ELSE!!!!!
        ###
        if real_data.shape[0] < real_PP.shape[0]:
            prediction = self.predict_long(real_data, real_PP, n_days, long_PP=True)
        else:
            prediction = self.predict_long(real_data, real_PP, n_days)
        return prediction

    def apply_PP_fit(self, real_data, PP_min, PP_max, loss_fn):
        """
        real_Data: should be a time series of shape (L, 1, input_size) with:
            L: sequence length, e.g. 50 days
            1: only one time series at a time can be fitted
            input_size: as above, generally 1
        PP_min: minimum values of possible pandemic parameters, tensor of shape (5) with:
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        PP_max: maximum values of possible pandemic parameters, tensor of shape (5) with:
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        """
        ####
        # FOR NOW; THIS ONLY ALLOWS A SINGLE PP TUPLE TO HOLD TRUE THE ENTIRE TIME
        # THIS MEANS 
        ####
        n_days = real_data.shape[0] - self.backtime
        N_range = np.arange(PP_min[0], PP_max[0], step=max(1, (PP_max[0]-PP_min[0])//10))
        D_range = np.arange(PP_min[1], PP_max[1], step=max(1, (PP_max[1]-PP_min[1])//10))
        r_range = np.arange(PP_min[2], PP_max[2], step=max(0.1, (PP_max[2]-PP_min[2])/10.0))
        d_range = np.arange(PP_min[3], PP_max[3], step=max(1, (PP_max[3]-PP_min[3])//10))
        epsilon_range = np.arange(PP_min[4], PP_max[4], step=max(0.1, (PP_max[4]-PP_min[4])/10.0))
        print(N_range)
        print(D_range)
        print(r_range)
        print(d_range)
        print(epsilon_range)
        current_loss = np.inf
        current_best = None
        for N in N_range:
            for D in D_range:
                for r in r_range:
                    for d in d_range:
                        for epsilon in epsilon_range:
                            PP_inp = torch.Tensor([N, D, r, d, epsilon]).repeat(real_data.shape[0], 1, 1)
                            ###
                            # Check if repetition works
                            # check if long_PP=True is necessary (AS ABOVE)
                            ###
                            prediction = self.predict_long(real_data[self.backtime:], PP_inp, n_days)
                            loss = loss_fn(prediction, real_data[:self.backtime])
                            if loss < current_loss:
                                current_loss = loss
                                current_best = torch.Tensor([N, D, r, d, epsilon])
        return current_best, current_loss



# Example use
if __name__ == "__main__":
    input_size = 1
    hidden_size = 256
    num_layers = 2
    dropout = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backtime = 20  # number of days the network gets to see before prediction
    foretime = 3  # number of days to predict for long predictions
    batch_length = 2500
    train_ratio = 0.7
    batch_size = 1
    DATA_PATH = "data_path.pt"
    PP_PATH = "PP_path.pt"
    mylstm = LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            foretime=foretime,
                            backtime=backtime)
    train_inds, test_inds = torch.utils.data.random_split(
        torch.arange(batch_length), 
        [int(batch_length*train_ratio), batch_length - int(batch_length*train_ratio)], 
        generator=torch.Generator().manual_seed(17))
    def my_collate(batch):
        x = torch.zeros(size=(backtime, len(batch), 1))
        pp = torch.zeros(size=(backtime, len(batch), 5))
        y = torch.zeros(size=(foretime, len(batch), 1))
        for i, item in enumerate(batch):
            x[:, i, :] = item[0]
            pp[:, i, :] = item[1]
            y[:, i, :] = item[2]
        return x, pp, y
    training_data = Dataset.Dataset(DATA_PATH, PP_PATH, train_inds, backtime=backtime, foretime=foretime)
    test_data = Dataset.Dataset(DATA_PATH, PP_PATH, test_inds, backtime=backtime, foretime=foretime)
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    loss_fn = torch.nn.MSELoss()