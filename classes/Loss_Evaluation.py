import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import LSTM_Model
import RNN_Model
import Dataset

data_version = 2
directory = "C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/data/Tuning_Logs/"
if data_version == 1:
    batch_length = 20
else:
    batch_length = 20000
backtime = 20
foretime = 3
DATA_PATH = f"C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/trainingdata/v{data_version}/data_v{data_version}.pt"
PP_PATH = f"C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/trainingdata/v{data_version}/pp_v{data_version}.pt"
batch_size = 2048
train_ratio = 0.7
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def get_losses():
    names = []
    losses = []
    for filename in os.listdir(directory):
        if filename[-4:] != ".txt":
            data = filename.split("-")
            name = data[0]
            hidden_size = int(data[1][-3:])
            num_layers = int(data[2][-1])
            if name == "RNN_Model":
                nonlinearity = data[3][13:]
                model = RNN_Model.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity)
            else:
                dropout = float(data[3][8:])
                model = LSTM_Model.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            model.load_model(directory+filename)
            model.to(device)
            loss = []
            for X, PP, y in test_dataloader:
                test_loss = model.test_model(test_X=X, test_PP=PP, test_y=y, loss_fn=loss_fn, n_days=foretime)
                loss.append(test_loss)
                losses.append(np.mean(loss))
                names.append(filename)
                break
    return names, losses

def plot_predictions(filename):
    data = filename.split("-")
    name = data[0]
    hidden_size = int(data[1][-3:])
    num_layers = int(data[2][-1])
    if "e" not in data[4][14:]:
        learning_rate = float(data[4][14:])
    else:
        learning_rate = float(data[4][14:] + data[5])
    if name == "RNN_Model":
        nonlinearity = data[3][13:]
        model = RNN_Model.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity)
    else:
        dropout = float(data[3][8:])
        model = LSTM_Model.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_model(directory+filename)
    X = 0
    PP = 0
    y = 0
    for i in range(5):
        for Xt, PPt, yt in test_dataloader:
            X = Xt[:, 0, :].view(-1, 1, 1)
            PP = PPt[:, 0, :].view(-1, 1, 5)
            y = yt[:, 0, :].view(-1, 1, 1)
            break
        model.to(device)
        pred = model.predict_long(X, PP, 50)
        X = X.view(-1)
        y = y.view(-1)
        pred = pred.view(-1).detach().numpy()

        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(len(X)), X, label="Input", color="C0")
        plt.scatter(np.arange(len(X), len(X)+len(y)), y,label="Truth", color="C0", marker="x")
        plt.plot(np.arange(len(X), len(X)+len(pred)), pred, label="Prediction", color="C1")
        # plt.ylim((-0.1, 1.1))
        plt.title(filename)
        plt.legend()
        plt.show()

def get_path(type, hidden, layers, spec, lr):
    if lr == 0.00001:
        lr_mod = 1e-05
    else:
        lr_mod = lr
    if type == "RNN":
        path = f"{type}_Model-hidden_size_{hidden}-num_layers_{layers}-nonlinearity_{spec}-learning_rate_{lr_mod}"
    elif type == "LSTM":
        path = f"{type}_Model-hidden_size_{hidden}-num_layers_{layers}-dropout_{spec}-learning_rate_{lr_mod}"
    return path

def get_best():
    names, losses = get_losses()
    best_rnn = None
    best_rnn_loss = np.inf
    best_lstm = None
    best_lstm_loss = np.inf
    for i, name in enumerate(names):
        if "RNN" in name:
            if losses[i] < best_rnn_loss:
                best_rnn = name
                best_rnn_loss = losses[i]
        elif "LSTM" in name:
            if losses[i] < best_lstm_loss:
                best_lstm = name
                best_lstm_loss = losses[i]
    return best_rnn, best_rnn_loss, best_lstm, best_lstm_loss

def compare_nonlinearity(names, losses):
    best_tanh = None
    best_tanh_loss = np.inf
    best_relu = None
    best_relu_loss = np.inf
    for i, name in enumerate(names):
        if "RNN" in name:
            if "tanh" in name:
                if losses[i] < best_tanh_loss:
                    best_tanh = name
                    best_tanh_loss = losses[i]
            elif "relu" in name:
                if losses[i] < best_relu_loss:
                    best_relu = name
                    best_relu_loss = losses[i]
    return best_relu, best_relu_loss, best_tanh, best_tanh_loss

def compare_dropout(names, losses):
    best_drop = None
    best_drop_loss = np.inf
    best_nodrop = None
    best_nodrop_loss = np.inf
    for i, name in enumerate(names):
        if "LSTM" in name:
            if "dropout_0.0" in name:
                if losses[i] < best_nodrop_loss:
                    best_nodrop = name
                    best_nodrop_loss = losses[i]
            elif "dropout_0.5" in name:
                if losses[i] < best_drop_loss:
                    best_drop = name
                    best_drop_loss = losses[i]
    return best_drop, best_drop_loss, best_nodrop, best_nodrop_loss

def compare_layers(names, losses):
    best_rnn_1 = None
    best_rnn_1_loss = np.inf
    best_rnn_2 = None
    best_rnn_2_loss = np.inf
    best_lstm_1 = None
    best_lstm_1_loss = np.inf
    best_lstm_2 = None
    best_lstm_2_loss = np.inf
    for i, name in enumerate(names):
        if "RNN" in name:
            if "num_layers_1" in name:
                if losses[i] < best_rnn_1_loss:
                    best_rnn_1 = name
                    best_rnn_1_loss = losses[i]
            elif "num_layers_2" in name:
                if losses[i] < best_rnn_2_loss:
                    best_rnn_2 = name
                    best_rnn_2_loss = losses[i]
        elif "LSTM" in name:
            if "num_layers_1" in name:
                if losses[i] < best_lstm_1_loss:
                    best_lstm_1 = name
                    best_lstm_1_loss = losses[i]
            elif "num_layers_2" in name:
                if losses[i] < best_lstm_2_loss:
                    best_lstm_2 = name
                    best_lstm_2_loss = losses[i]
    return best_rnn_1, best_rnn_1_loss, best_rnn_2, best_rnn_2_loss, best_lstm_1, best_lstm_1_loss, best_lstm_2, best_lstm_2_loss

def compare_hidden(names, losses):
    best_rnn_128 = None
    best_rnn_128_loss = np.inf
    best_rnn_256 = None
    best_rnn_256_loss = np.inf
    best_rnn_512 = None
    best_rnn_512_loss = np.inf
    best_lstm_128 = None
    best_lstm_128_loss = np.inf
    best_lstm_256 = None
    best_lstm_256_loss = np.inf
    best_lstm_512 = None
    best_lstm_512_loss = np.inf
    for i, name in enumerate(names):
        if "RNN" in name:
            if "hidden_size_128" in name:
                if losses[i] < best_rnn_128_loss:
                    best_rnn_128 = name
                    best_rnn_128_loss = losses[i]
            elif "hidden_size_256" in name:
                if losses[i] < best_rnn_256_loss:
                    best_rnn_256 = name
                    best_rnn_256_loss = losses[i]
            elif "hidden_size_512" in name:
                if losses[i] < best_rnn_512_loss:
                    best_rnn_512 = name
                    best_rnn_512_loss = losses[i]
        elif "LSTM" in name:
            if "hidden_size_128" in name:
                if losses[i] < best_lstm_128_loss:
                    best_lstm_128 = name
                    best_lstm_128_loss = losses[i]
            elif "hidden_size_256" in name:
                if losses[i] < best_lstm_256_loss:
                    best_lstm_256 = name
                    best_lstm_256_loss = losses[i]
            elif "hidden_size_512" in name:
                if losses[i] < best_lstm_512_loss:
                    best_lstm_512 = name
                    best_lstm_512_loss = losses[i]
    return best_rnn_128, best_rnn_128_loss, best_rnn_256, best_rnn_256_loss, best_rnn_512, best_rnn_512_loss, best_lstm_128, best_lstm_128_loss, best_lstm_256, best_lstm_256_loss, best_lstm_512, best_lstm_512_loss

def compare_lrs(names, losses):
    best_rnn_01 = None
    best_rnn_01_loss = np.inf
    best_rnn_001 = None
    best_rnn_001_loss = np.inf
    best_rnn_0001 = None
    best_rnn_0001_loss = np.inf
    best_rnn_00001 = None
    best_rnn_00001_loss = np.inf
    best_lstm_01 = None
    best_lstm_01_loss = np.inf
    best_lstm_001 = None
    best_lstm_001_loss = np.inf
    best_lstm_0001 = None
    best_lstm_0001_loss = np.inf
    best_lstm_00001 = None
    best_lstm_00001_loss = np.inf
    for i, name in enumerate(names):
        if "RNN" in name:
            if "learning_rate_0.01" in name:
                if losses[i] < best_rnn_01_loss:
                    best_rnn_01 = name
                    best_rnn_01_loss = losses[i]
            elif "learning_rate_0.001" in name:
                if losses[i] < best_rnn_001_loss:
                    best_rnn_001 = name
                    best_rnn_001_loss = losses[i]
            elif "learning_rate_0.0001" in name:
                if losses[i] < best_rnn_0001_loss:
                    best_rnn_0001 = name
                    best_rnn_0001_loss = losses[i]
            elif "learning_rate_1e-5" in name:
                if losses[i] < best_rnn_00001_loss:
                    best_rnn_00001 = name
                    best_rnn_00001_loss = losses[i]
        elif "LSTM" in name:
            if "learning_rate_0.01" in name:
                if losses[i] < best_lstm_01_loss:
                    best_lstm_01 = name
                    best_lstm_01_loss = losses[i]
            elif "learning_rate_0.001" in name:
                if losses[i] < best_lstm_001_loss:
                    best_lstm_001 = name
                    best_lstm_001_loss = losses[i]
            elif "learning_rate_0.0001" in name:
                if losses[i] < best_lstm_0001_loss:
                    best_lstm_0001 = name
                    best_lstm_0001_loss = losses[i]
            elif "learning_rate_1e-5" in name:
                if losses[i] < best_lstm_00001_loss:
                    best_lstm_00001 = name
                    best_lstm_00001_loss = losses[i]
    return best_rnn_01, best_rnn_01_loss, best_rnn_001, best_rnn_001_loss, best_rnn_0001, best_rnn_0001_loss, best_rnn_00001, best_rnn_00001_loss, best_lstm_01, best_lstm_01_loss, best_lstm_001, best_lstm_001_loss, best_lstm_0001, best_lstm_0001_loss, best_lstm_00001, best_lstm_00001_loss

def run_comparison():
    names, losses = get_losses()
    print("working..")
    best_relu, best_relu_loss, best_tanh, best_tanh_loss = compare_nonlinearity()
    print("working..")
    best_drop, best_drop_loss, best_nodrop, best_nodrop_loss = compare_dropout()
    print("working..")
    best_rnn_1, best_rnn_1_loss, best_rnn_2, best_rnn_2_loss, best_lstm_1, best_lstm_1_loss, best_lstm_2, best_lstm_2_loss = compare_layers()
    print("working..")
    best_rnn_128, best_rnn_128_loss, best_rnn_256, best_rnn_256_loss, best_rnn_512, best_rnn_512_loss, best_lstm_128, best_lstm_128_loss, best_lstm_256, best_lstm_256_loss, best_lstm_512, best_lstm_512_loss = compare_hidden()
    print("working..")
    best_rnn_01, best_rnn_01_loss, best_rnn_001, best_rnn_001_loss, best_rnn_0001, best_rnn_0001_loss, best_rnn_00001, best_rnn_00001_loss, best_lstm_01, best_lstm_01_loss, best_lstm_001, best_lstm_001_loss, best_lstm_0001, best_lstm_0001_loss, best_lstm_00001, best_lstm_00001_loss = compare_lrs()

    print(f"AA & BB")
    print(f"\hline")
    print(f"a & b")
    print(f"c & d)")

run_comparison()