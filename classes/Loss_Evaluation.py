import os
from pickle import UnpicklingError
import torch
import numpy as np
import matplotlib.pyplot as plt
import LSTM_Model
import RNN_Model
import Dataset

directory = "C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/data/Tuning_Logs/"
batch_length = 250
backtime = 20
foretime = 3
DATA_PATH = "C:/Users/gooog/Desktop/AML/final_project/repository/data_path.pt"
PP_PATH = "C:/Users/gooog/Desktop/AML/final_project/repository/PP_path.pt"
batch_size = 1
train_ratio = 0.7
device = "cuda" if torch.cuda.is_available() else "cpu"

files = []
for filename in os.listdir(directory):
    files.append(filename)

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

def compare_all():
    best = np.inf
    best_name = ""
    plt.figure(figsize=(16, 16))
    i = 1
    for filename in os.listdir(directory):
        if filename[-4:] == ".txt":
            arr = np.loadtxt(directory+filename)
            if len(arr) == 0:
                continue
            plt.subplot(8, 8, i) 
            plt.plot(arr)
            # plt.ylim((-0.1, 0.2))
            if i == -16: 
                break
            i += 1
            if np.mean(arr[-10:]) < best:
                best = np.mean(arr[-10:])
                best_name = filename
    plt.show()
    print(best, best_name)

def test_model(filename):
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
    for Xt, PPt, yt in test_dataloader:
        X = Xt
        PP = PPt
        y = yt
        break
    model.to(device)
    pred = model.predict_long(X, PP, 3)
    pred2 = model.forward_long(X, PP, 3)
    X = X.view(-1)
    PP = PP.view(-1)
    y = y.view(-1)
    pred = pred.view(-1).detach().numpy()
    pred2 = pred2.view(-1).detach().numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(X)), X, label="Input", color="C0")
    plt.scatter(np.arange(len(X), len(X)+len(y)), y,label="Truth", color="C0", marker="x")
    plt.plot(np.arange(len(X), len(X)+len(pred)), pred, label="Prediction", color="C1")
    plt.scatter(np.arange(len(X), len(X)+len(pred2)), pred2, label="Prediction 2", color="C2", marker="x")
    # plt.ylim((-0.1, 1.1))
    plt.title(filename)
    plt.legend()
    plt.show()

for filename in os.listdir(directory):
        if filename[-4:] != ".txt":
            test_model(filename)

compare_all()