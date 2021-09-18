import torch
from torch import nn
from Datahandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np

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
        self.input_size = input_size + 5
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        self.get_h0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)
        self.get_c0 = nn.Linear(in_features=5, out_features=self.num_layers * self.hidden_size)

    def forward(self, sequence, inits=None):
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
        preds = []
        L = sequence.shape[0] # 19,1,1
        for i in range(n_days):
            PP_sequence = torch.cat((sequence, PP_input), dim=2) #19,1,6 (number of days, ..., 6 dimensions [5PP,1value]); 10,1,6
            self.eval()
            with torch.no_grad():
                # Compute prediction error
                h_0 = self.get_h0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                c_0 = self.get_c0(PP_input[0]).view(self.num_layers, PP_input.shape[1], self.hidden_size) #2,1,256
                pred = self.forward(PP_sequence, inits=(h_0, c_0))
                preds.append(pred) #1,1,1
                sequence = torch.cat((sequence, pred), dim=0)[-n_days:]#19,1,1 -> 20,1,1 -> 10,1,1 -> select last 10 steps; basically move one step forward with the predicted value 
                PP_input = torch.cat((PP_input, PP_input[-1].view(1, -1, 5)), dim=0)[-n_days:] # ;11,1,5 -> 10,1,5
                #TODO append prediction for PP_input or do they remain the same?
        return torch.Tensor(preds)

    def train_model(self, training_data, training_PP, loss_fn, optimizer, verbose=False):
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
        X_data = training_data[:-self.output_size]
        y_data = training_data[-self.output_size:]
        X_PP = training_PP[:-self.output_size]

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
            c_0 = self.get_c0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
            pred = self.forward(test_seq, inits=(h_0, c_0))
            test_loss = loss_fn(pred, y_data).item()
        print("Average Test Loss:", test_loss)

        return test_loss
    
    '''
    def test_model_long(self, test_data, test_PP, loss_fn, n_days):
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
        # divide into training (X) and validation set 
        test_data = test_data.to(device)
        test_PP = test_PP.to(device)
        X_data = test_data[:-n_days]
        y_data = test_data[-n_days:]
        X_PP = test_PP[:-n_days]
        #test_seq = torch.cat((X_data, X_PP), dim=2)
        
        test_loss = 0

        self.eval()
        with torch.no_grad():
            # Compute prediction error
            h_0 = self.get_h0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
            c_0 = self.get_c0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
            pred = self.forward(test_seq, inits=(h_0, c_0))
            test_loss = loss_fn(pred, y_data).item()
        print("Average Test Loss:", test_loss)
        return torch.Tensor(preds)

        preds = []
        L = test_seq.shape[0]
        for i in range(n_days):
            PP_sequence = torch.cat((X_data, X_PP), dim=2)
            self.eval()
            with torch.no_grad():
                # Compute prediction error
                h_0 = self.get_h0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
                c_0 = self.get_c0(X_PP[0]).view(self.num_layers, X_PP.shape[1], self.hidden_size)
                pred = self.forward(PP_sequence, inits=(h_0, c_0))
                preds.append(pred)
                PP_sequence = torch.cat((PP_sequence, pred), dim=0)[-n_days:]
                X_PP = torch.cat((X_PP, X_PP[-1].view(1, -1, 5)), dim=0)[-n_days:]
        
        test_loss = loss_fn(preds, y_data).item()

    return test_loss
    '''
        

# Example use
if __name__ == "__main__":

    print("Running on:", device)

    B = 500  # batch size
    L = 20  # sequence length

    params = {"N": 100,
            "D": 10,
            "r": 0.2,
            "d": 14,
            "N_init": 1,
            "epsilon": 0.4,
            "version": "V2",
            "T": L + 1}
            
    DH = DataHandler(mode="Simulation", params=params, device=device)
    data, starting_points, PP_data  = DH(B,L)
    print("Data:            ", data.shape) #20,500,1
    training_data = data[:,:350,...]
    test_data = data[:,350:,...]
    print("Training Data:   ", training_data.shape) #20,350,1
    print("Test Data:       ", test_data.shape) #20,150,1

    print("PP Data:         ", PP_data.shape) #20,500,5
    PP_training_data = PP_data[:,:350,...]
    PP_test_data = PP_data[:,350:,...]
    print("PP Training Data:", PP_training_data.shape) #20,350,5
    print("PP Test Data:    ", PP_test_data.shape) #20,150,5

    input_size = 1
    hidden_size = 256
    output_size = 1
    num_layers = 2
    dropout = 0.5

    n_epochs = 2
    learning_rate = 0.0001

    n_days = 10


    MyLSTM = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)

    # Print model and its parameters

    for name, param in MyLSTM.named_parameters():
        if param.requires_grad:
            print(name)


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=MyLSTM.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        MyLSTM.train_model(training_data=training_data,training_PP=PP_training_data, loss_fn=loss_fn, optimizer=optimizer, verbose=False)
        if epoch % 100 == 0:
            MyLSTM.test_model(test_data=test_data, test_PP=PP_test_data, loss_fn=loss_fn)

    # get correct samples 

    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        test_slice_X = test_data[:,i,...].view(L, 1, -1)[:L-n_days]
        test_slice_y = test_data[:,i,...].view(L, 1, -1)[L-n_days:].to("cpu").view(-1).detach().numpy()
        PP_test_slice = PP_test_data[:,i,...].view(L, 1, 5)[:L-n_days].to(device)
        pred = MyLSTM.predict_long(test_slice_X, PP_test_slice, n_days=n_days).to("cpu").view(-1).detach().numpy()
        plt.plot(np.arange(L-n_days), test_slice_X.to("cpu").view(-1).detach().numpy(), color="C0", label="Test Set")
        plt.scatter(np.arange(L-n_days,L), pred, color="C1", label="prediction")
        plt.scatter(np.arange(L-n_days,L), test_slice_y, color="C0", marker="x", label="ground truth")
        plt.legend()
    plt.savefig('plots/lstm4.jpg')
