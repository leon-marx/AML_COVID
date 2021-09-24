# This file tests, wether the RNN and LSTM classes work as expected using data generated by the Sampler class.

# Package Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

# Own Imports
import Dataset
import LSTM_Model
import RNN_Model

# Parameters
input_size = 1
hidden_size = 256
num_layers = 2
nonlinearity = "tanh"
dropout = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 100

learning_rate = 0.0001
backtime = 20  # number of days the network gets to see before prediction
foretime = 3  # number of days to predict for long predictions
batch_length = 2500
train_ratio = 0.7
batch_size = 1024
DATA_PATH = "data_path.pt"
PP_PATH = "PP_path.pt"

# Models
print("Initializing Models")
myrnn = RNN_Model.RNN(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      nonlinearity=nonlinearity,
                      foretime=foretime,
                      backtime=backtime)


mylstm = LSTM_Model.LSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         foretime=foretime,
                         backtime=backtime)
# Dataloader
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

def training_loop(model, name):
    # Training
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    t_loop = tqdm.trange(n_epochs, leave=True)
    for epoch in t_loop:
        for X, X_PP, y in training_dataloader:
            train_loss = model.train_model(training_X=X, training_PP=X_PP, training_y=y, loss_fn=loss_fn, optimizer=optimizer)
            t_loop.set_description(f"Epoch: {epoch}, Training Loss: {train_loss}")
        if epoch % 100 == 0:
            test_losses = []
            for X_t, X_PP_t, y_t in test_dataloader:
                test_loss = model.test_model(test_X=X_t, test_PP=X_PP_t, test_y=y_t, loss_fn=loss_fn)
                test_losses.append(test_loss)
            print(f"Average Test Loss: {np.mean(test_losses)}")

    # Plotting Long Predictions
    for X, X_PP, y in test_dataloader:
        plot_X = X[:, :4, :]
        plot_PP = X_PP[:, :4, :]
        plot_y = y[:, :4, :]
        break
    print(f"Plotting Long {name} Predictions")
    x = np.arange(backtime)
    y = np.arange(backtime, backtime+foretime)
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        test_slice_X = plot_X[:, i, :].view(-1, 1, 1)
        test_slice_y = plot_y[:, i, :].view(-1, 1, 1).to("cpu").view(-1).detach().numpy()
        PP_test_slice = plot_PP[:, i, :].view(-1, 1, 5).to(device)
        preds = model.predict_long(test_slice_X, PP_test_slice, n_days=foretime).to("cpu").view(-1).detach().numpy()
        plt.plot(x, test_slice_X.to("cpu").view(-1), color="C0", label="Test Set")
        plt.scatter(y, preds, color="C1", label="Prediction")
        plt.scatter(y, test_slice_y, color="C0", marker="x", label="Truth")
        plt.ylim((-0.1, 1.1))
        plt.legend()
    plt.savefig(f"{name}_Long_Predictions.png")
    torch.save(mylstm.state_dict(), f"Trained_{name}_Model")

# LSTM Training
print("Training LSTM")
mylstm.to(device)
# training_loop(mylstm, "LSTM")

# RNN Training
print("Training RNN")
myrnn.to(device)
training_loop(myrnn, "RNN")
training_loop(myrnn, "RNN")
training_loop(myrnn, "RNN")
training_loop(myrnn, "RNN")
training_loop(myrnn, "RNN")
