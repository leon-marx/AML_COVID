# Package Imports
import numpy as np
import torch
import tqdm

# Own Imports
import Dataset
import RNN_Model
import LSTM_Model

# Fixed Parameters
input_size = 1
DATA_PATH = "data_path.pt"
PP_PATH = "pp_path.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
foretime = 3
backtime = 20
n_epochs = 1000
LOG_FOLDER = "Tuning_Logs"
batch_length = 2500
train_ratio = 0.7
batch_size = 2048
random_seed = 17

# Parameters to Tune
hidden_size_list = [128, 256, 512]
num_layers_list = [1, 2]
dropout_list = [0.0, 0.5]
nonlinearity_list = ["tanh", "relu"]
learning_rate_list = [0.00001, 0.0001, 0.001, 0.01]

# Dataloader
train_inds, test_inds = torch.utils.data.random_split(
    torch.arange(batch_length), 
    [int(batch_length*train_ratio), batch_length - int(batch_length*train_ratio)], 
    generator=torch.Generator().manual_seed(random_seed))

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

def tune_rnn(hidden_size, num_layers, nonlinearity, learning_rate):
    # Model
    print("Initializing Model with:")
    print(f"    hidden_size: {hidden_size}")
    print(f"    num_layers: {num_layers}")
    print(f"    nonlinearity: {nonlinearity}")
    print(f"    learning_rate: {learning_rate}")
    myrnn = RNN_Model.RNN(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity=nonlinearity,
                            foretime=foretime,
                            backtime=backtime)
    # RNN Training
    print("Training RNN")
    myrnn.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=myrnn.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    t_loop = tqdm.trange(n_epochs, leave=True)
    for epoch in t_loop:
        for X, X_PP, y in training_dataloader:
            train_loss = myrnn.train_model(training_X=X, training_PP=X_PP, training_y=y, loss_fn=loss_fn, optimizer=optimizer)
            train_losses.append(train_loss)
            t_loop.set_description(f"Epoch: {epoch}, Training Loss: {train_loss}")
        if epoch % 100 == 0:
            for X_t, X_PP_t, y_t in test_dataloader:
                test_loss = myrnn.test_model(test_X=X_t, test_PP=X_PP_t, test_y=y_t, loss_fn=loss_fn)
                test_losses.append(test_loss)
            print(f"Average Test Loss: {np.mean(test_losses)}")
    with open(f"{LOG_FOLDER}/RNN-hidden_size_{hidden_size}-num_layers_{num_layers}-nonlinearity_{nonlinearity}-learning_rate_{learning_rate}_train.txt", "w") as file:
        for loss in train_losses:
            file.write(str(loss) + "\n")
    with open(f"{LOG_FOLDER}/RNN-hidden_size_{hidden_size}-num_layers_{num_layers}-nonlinearity_{nonlinearity}-learning_rate_{learning_rate}_test.txt", "w") as file:
        for loss in test_losses:
            file.write(str(loss) + "\n")
    torch.save(myrnn.state_dict(), f"{LOG_FOLDER}/RNN_Model-hidden_size_{hidden_size}-num_layers_{num_layers}-nonlinearity_{nonlinearity}-learning_rate_{learning_rate}")
    print("")
    print("")
    print("")

def tune_lstm(hidden_size, num_layers, dropout, learning_rate):
    # Model
    print("Initializing Model with:")
    print(f"    hidden_size: {hidden_size}")
    print(f"    num_layers: {num_layers}")
    print(f"    dropout: {dropout}")
    print(f"    learning_rate: {learning_rate}")
    mylstm = LSTM_Model.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            foretime=foretime,
                            backtime=backtime)
    # LSTM Training
    print("Training LSTM")
    mylstm.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=mylstm.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    t_loop = tqdm.trange(n_epochs, leave=True)
    for epoch in t_loop:
        for X, X_PP, y in training_dataloader:
            train_loss = mylstm.train_model(training_X=X, training_PP=X_PP, training_y=y, loss_fn=loss_fn, optimizer=optimizer)
            t_loop.set_description(f"Epoch: {epoch}, Training Loss: {train_loss}")
        if epoch % 100 == 0:
            test_losses = []
            for X_t, X_PP_t, y_t in test_dataloader:
                test_loss = mylstm.test_model(test_X=X_t, test_PP=X_PP_t, test_y=y_t, loss_fn=loss_fn)
                test_losses.append(test_loss)
            print(f"Average Test Loss: {np.mean(test_losses)}")
    with open(f"{LOG_FOLDER}/LSTM-hidden_size_{hidden_size}-num_layers_{num_layers}-dropout_{dropout}-learning_rate_{learning_rate}_train.txt", "w") as file:
        for loss in train_losses:
            file.write(str(loss) + "\n")
    with open(f"{LOG_FOLDER}/LSTM-hidden_size_{hidden_size}-num_layers_{num_layers}-dropout_{dropout}-learning_rate_{learning_rate}_test.txt", "w") as file:
        for loss in test_losses:
            file.write(str(loss) + "\n")
    torch.save(mylstm.state_dict(), f"{LOG_FOLDER}/LSTM_Model-hidden_size_{hidden_size}-num_layers_{num_layers}-dropout_{dropout}-learning_rate_{learning_rate}")
    print("")
    print("")
    print("")

print("Running Loop")
for hidden_size in hidden_size_list:
    for num_layers in num_layers_list:
        for learning_rate in learning_rate_list:
            for nonlinearity in nonlinearity_list:
                try:
                    tune_rnn(hidden_size, num_layers, nonlinearity, learning_rate)
                except RuntimeError:
                    print("A step was skipped due to a RuntimeError!")
            for dropout in dropout_list:
                try:
                    tune_lstm(hidden_size, num_layers, dropout, learning_rate)
                except RuntimeError:
                    print("A step was skipped due to a RuntimeError!")
