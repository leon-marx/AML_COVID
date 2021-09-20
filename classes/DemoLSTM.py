import torch
from torch import nn
import torch.optim as optim
from torch.nn.modules.linear import Linear
from Datahandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np


class DemoLSTM(nn.Module):
    """LSTM for time series prediction"""

    def __init__(self, n_hidden=51):
        super(DemoLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        # N (batch_size), 100 (samples)

        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        # if only one one timestep into the future: each one timestep enters the LSTM over all timesteps including the last one for the future prediction
        for input_t in x.split(1, dim=1):
            # N (batch_size), 1 (sample)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        # if #future timesteps
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)            

        outputs = torch.cat(outputs, dim=1)
        
        return outputs



if __name__ == "__main__":
    
    # Define device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # Define params
    split = 0.2
    B = 10  # batch size; number of time series 
    L = 20  # sequence length
    pred_steps = 2 # steps to predict
    future = 1 # needs to be equal to pred_steps

    output_size=5
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
    
    # shifted train input and target 
    training_data = data[:,:7,...]
    test_data = data[:,7:,...]
    train_input = training_data[:-1].transpose(0,1).squeeze()
    train_target = training_data[1:].transpose(0,1).squeeze()
    test_input = test_data[:-pred_steps].transpose(0,1).squeeze()
    test_target = test_data[1:].transpose(0,1).squeeze()
    print("Data:", data.shape) #20,500,1
    print("Training Data: ", training_data.shape) #20,350,1
    print("Test Data:", test_data.shape) #20,150,1

    model = DemoLSTM(n_hidden=51)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 50

    for i in range(n_steps):
        print(f"Step {i}")

        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print(f"loss: {loss.item()}")
            loss.backward()
            return loss 
        
        optimizer.step(closure)

        with torch.no_grad():
            pred = model(test_input, future=future)
            loss = criterion(pred, test_target) #pred[:,:-future], test_target)
            print(f"test loss: {loss.item()}")
            test_pred = pred.detach().numpy()
    
        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        
        def draw(pred, gt):
            plt.plot(np.arange(gt.size(0)), gt, 'b', linewidth=2.0)
            plt.plot(np.arange(pred.size), pred, 'r' + ":", linewidth=2.0)

        draw(test_pred[0], test_input[0])

        plt.savefig("plots/lstm/predict%d.pdf"%i)
        plt.close()

        

