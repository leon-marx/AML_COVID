# This is the main file which runs the whole Game Plan

# Package Imports
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
import torch

# Own Imports
from LSTM_Model import LSTM
from Datahandler import Sampler



class Pipeline():
    
    def __init__(self, N, K, T, B, L, version, train_test_split, timesteps_to_predict, num_epochs, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, device="cuda" if torch.cuda.is_available() else "cpu") -> None:
        
        # Define device 
        self.device = device 

        # Hyperparameters for simulation 
        self.N = N 
        self.K = K
        self.T = T
        self.B = B #3 # batch size -> how many sequences should be sampled 
        self.L = L #10
        self.version = version
        self.device = device
        self.train_test_split = train_test_split

        # Hyperparameters for network
        self.timesteps_to_predict = timesteps_to_predict
        self.epochs = num_epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

    def execute(self):

        print(f"Started Pipeline | Running on: {self.device}")

        # Get sampled time-series from Simulation based on optimized PP's
        S = Sampler(lower_lims=None, upper_lims=None, N=N, T=T, version="V2", device="cpu")
        data,PP_data,starting_points = S(K=K,L=L,B=B,mode="optimized")

        # Split training and test data 
        training_data = data[:int(np.floor(B*K*(1-train_test_split))),...].transpose(0,1).unsqueeze(dim=2) #TODO typically batch-size at index 0?
        test_data = data[int(np.ceil(B*K*(1-train_test_split))):,...].transpose(0,1).unsqueeze(dim=2)
        PP_training_data = PP_data[:int(np.floor(B*K*(1-train_test_split))),...].repeat(L,1,1)
        PP_test_data = PP_data[int(np.ceil(B*K*(1-train_test_split))):,...].repeat(L,1,1)
        print("Data:", data.shape)                              #L,B,1
        print("Training Data:   ", training_data.shape)         #L,B*train_test_split,1
        print("Test Data:       ", test_data.shape)             #L,B*train_test_split,1
        print("PP Data:         ", PP_data.shape)               #L,B,5
        print("PP Training Data:", PP_training_data.shape)      #L,B*train_test_split,5
        print("PP Test Data:    ", PP_test_data.shape)          #L,B*train_test_split,5

        # Network Training
        MyLSTM = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.output_size, num_layers=self.num_layers, dropout=self.dropout).to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(params=MyLSTM.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            MyLSTM.train_model(training_data=training_data,training_PP=PP_training_data, loss_fn=loss_fn, optimizer=optimizer, verbose=True, epoch=epoch)
            if epoch % 20 == 0:
                MyLSTM.test_model(test_data=test_data, test_PP=PP_test_data, loss_fn=loss_fn)
        
        # Plot results 
        plt.figure(figsize=(12, 12))
        for i in range(test_data.shape[1]):
            plt.subplot(3, 3, i+1)
            test_slice_X = test_data[:,i,...].view(L, 1, -1)[:L-timesteps_to_predict]
            test_slice_y = test_data[:,i,...].view(L, 1, -1)[L-timesteps_to_predict:].to("cpu").view(-1).detach().numpy()
            PP_test_slice = PP_test_data[:,i,...].view(L, 1, 5)[:L-timesteps_to_predict].to(device)
            pred = MyLSTM.predict_long(test_slice_X, PP_test_slice, n_days=timesteps_to_predict).to("cpu").view(-1).detach().numpy()
            plt.plot(np.arange(L-timesteps_to_predict), test_slice_X.to("cpu").view(-1).detach().numpy(), color="C0", label="Test Set")
            plt.scatter(np.arange(L-timesteps_to_predict,L), pred, color="C1", label="prediction")
            plt.scatter(np.arange(L-timesteps_to_predict,L), test_slice_y, color="C0", marker="x", label="ground truth")
            plt.legend()
        plt.savefig('plots/lstm5.jpg')

if __name__ == '__main__':

    # Hyperparameters for simulation 
    N = 1000 
    K = 10 # how many different simulation samples 
    T = 50
    B = 1 #3 # batch size -> how many sequences should be sampled #TODO Why the same data multiple times?
    L = 20 #10
    version="V2"
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_test_split = 0.2

    # Hyperparameters for LSTM
    input_size = 1
    hidden_size = 256
    output_size = 1
    num_layers = 2
    dropout = 0.5
    learning_rate = 0.0001

    num_epochs = 10
    timesteps_to_predict = 2

    # Start evaluation
    pipeline = Pipeline(N=N, K=K, T=T, B=B, L=L, version=version, train_test_split=train_test_split, timesteps_to_predict=timesteps_to_predict, num_epochs=num_epochs, input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, device="cuda" if torch.cuda.is_available() else "cpu")
    pipeline.execute()

    