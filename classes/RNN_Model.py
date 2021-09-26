import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import Dataset
import Datahandler
from tqdm import tqdm

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
            L_seq = sequence.shape[0]
            L_p = PP_input.shape[0] - L_seq
            # 19,1,6 (number of days, ..., 6 dimensions [5PP,1value]); 10,1,6
            PP_sequence = torch.cat((sequence, PP_input[:L_seq]), dim=2)            
            self.eval()
            with torch.no_grad():
                PP_ind = min(L_seq+L_p-1, L_seq+i)                
                if i == 0:
                    h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                    pred, h_final = self.forward(PP_sequence, h_0=h_0, full=True)
                    new_PP_seq = torch.cat((pred, PP_input[PP_ind].view(1, PP_input.shape[1], 5)), dim=2)
                    preds[i] = pred
                else:
                    pred, h_final = self.forward(new_PP_seq, h_final, full=True)
                    new_PP_seq = torch.cat((pred, PP_input[PP_ind].view(1, PP_input.shape[1], 5)), dim=2)
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
        self.load_state_dict(torch.load(path))

    def apply_PP_fit(self, real_data, PP_min, PP_max, PP_step, loss_fn):
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
        PP_step: step of the ranges of values of possible pandemic parameters, tensor of shape (5) with:
            5: the 5 different PP-values:
                N_pop: population size
                D: average degree of social network in population
                r: daily transmission rate between two individuals which were in contact
                d: duration of the infection
                epsilon: rate of cross-contacts
        """
        n_days = real_data.shape[0] - self.backtime
        N_range = np.arange(PP_min[0], PP_max[0], step=PP_step[0])
        D_range = np.arange(PP_min[1], PP_max[1], step=PP_step[1])
        r_range = np.arange(PP_min[2], PP_max[2], step=PP_step[2])
        d_range = np.arange(PP_min[3], PP_max[3], step=PP_step[3])
        epsilon_range = np.arange(PP_min[4], PP_max[4], step=PP_step[4])
        print("N_range", N_range)
        print("D_range", D_range)
        print("r_range", r_range)
        print("d_range", d_range)
        print("epsilon_range", epsilon_range)
        print("")
        current_loss = np.inf
        current_best = None
        for N in tqdm(N_range):
            for D in tqdm(D_range, leave=False):
                for r in tqdm(r_range, leave=False):
                    for d in tqdm(d_range, leave=False):
                        for epsilon in tqdm(epsilon_range, leave=False):
                            PP_inp = torch.Tensor([N, D, r, d, epsilon]).repeat(real_data.shape[0], 1, 1)
                            ###
                            # Check if repetition works
                            # check if long_PP=True is necessary (AS ABOVE)
                            ###
                            prediction = self.predict_long(real_data[:self.backtime], PP_inp, n_days).to(device)
                            loss = loss_fn(prediction, real_data[self.backtime:])
                            if loss < current_loss:
                                current_loss = loss
                                current_best = torch.Tensor([N, D, r, d, epsilon])
        return current_best, current_loss


# Example use
if __name__ == "__main__":
    input_size = 1
    hidden_size = 256
    num_layers = 2
    nonlinearity = "tanh"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backtime = 10  # number of days the network gets to see before prediction
    foretime = 3  # number of days to predict for long predictions
    batch_length = 250
    train_ratio = 0.7
    batch_size = 1
    DATA_PATH = "data_path.pt"
    PP_PATH = "PP_path.pt"
    PP_min = torch.Tensor([1, 1, 0.01, 1, 0.01])
    PP_max = torch.Tensor([10, 10, 0.8, 20, 0.8])
    PP_step = torch.Tensor([2, 1, 0.05, 3, 0.05])

    myrnn = RNN(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  nonlinearity=nonlinearity,
                  foretime=foretime,
                  backtime=backtime)

    myrnn.load_model(path="Training_Logs/RNN-hidden_size_256-num_layers_2-nonlinearity_tanh-learning_rate_0.0001.pt")

    myrnn.to(device)

    loss_fn = torch.nn.MSELoss()

    train_inds, test_inds = torch.utils.data.random_split(torch.arange(batch_length), [int(batch_length*train_ratio), batch_length-int(batch_length*train_ratio)], generator=torch.Generator().manual_seed(17))

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

    for X, PP, y in test_dataloader:
        X = X
        PP = PP
        y = y
        break

    pred = myrnn.forward_long(X, PP, 50)
    X = X.view(-1)
    PP = PP.view(-1)
    y = y.view(-1)
    pred = pred.view(-1).detach().numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(X)), X, label="Input", color="C0")
    plt.scatter(np.arange(len(X), len(X)+len(y)), y,label="Truth", color="C0", marker="x")
    plt.plot(np.arange(len(X), len(X)+len(pred)), pred, label="Prediction", color="C1")
    plt.ylim((-0.1, 1.1))
    plt.legend()
    plt.show()

    params_real = {"file": "Israel.txt", "wave": 3, "full": True, "use_running_average": False, "dt_running_average": 14}

    DH = Datahandler.DataHandler("Real", params_real, device=device)
    fit_data, starting_points = DH(B=None, L=None, return_plain=True)

    fitting_PP, fitting_loss = myrnn.apply_PP_fit(fit_data.view(fit_data.shape[0], 1, 1).type(torch.float32), PP_min, PP_max, PP_step, loss_fn)
    print(fitting_PP)
    print(fitting_loss)
