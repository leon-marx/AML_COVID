import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import LSTM_Model
import RNN_Model
import Dataset
import Datahandler

directory = "C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/data/Evaluation/"

device = "cuda" if torch.cuda.is_available() else "cpu"

best_rnn = "RNN_Model-hidden_size_128-num_layers_1-nonlinearity_tanh-learning_rate_1e-05"
best_lstm = "LSTM_Model-hidden_size_128-num_layers_1-dropout_0.0-learning_rate_0.0001"

lstm_PP = torch.load(directory + "NN_PP_fit/fitted_PPs_LSTM_Model")
rnn_PP = torch.load(directory + "NN_PP_fit/fitted_PPs_RNN_Model")
lstm_PP_loss = torch.load(directory + "NN_PP_fit/fitting_loss_LSTM_Model")
rnn_PP_loss = torch.load(directory + "NN_PP_fit/fitting_loss_RNN_Model")

print(rnn_PP, rnn_PP_loss)
print(lstm_PP, lstm_PP_loss)

def evaluate_PP_fit(filename, PP):
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
    params_real = {"file": "Israel.txt", "wave": 4, "full": False, "use_running_average": True, "dt_running_average": 14}
    DH = Datahandler.DataHandler("Real", params_real, device=device)
    fit_data, starting_points = DH(B=None, L=None, return_plain=True)
    fit_data = fit_data.view(-1, 1, 1).type(torch.float32)
    fit_data.to(device)
    PP = PP.repeat(fit_data.shape[0], 1, 1)
    X = fit_data[:40]
    print(len(X))
    y = fit_data[40:]
    pred = model.predict_long(X, PP, fit_data.shape[0]-40)
    pred = pred.view(-1).detach().numpy()
    X = X.view(-1).cpu()
    y = y.view(-1).cpu()

    
    params_simulation = {'N': 2500, 'version': 'V2', 'd': 5.5, 'D': 8, 'r': 0.03, 'r_new': 0.04, 'D_new': 9, 'T_change_D': 136, 'Smooth_transition': 10000.0, 'N_init': 10, 'T': 136, 'epsilon': 0.01}
    print(params_simulation["N_init"])
    DH = Datahandler.DataHandler(mode="Simulation", params=params_simulation, device=device)
    batches = []
    for i in range(5):
        batch, starting_points = DH(B=None, L=None, return_plain=True)
        batches.append(batch.cpu().numpy())
    batch = np.mean(batches, axis=0)

    plt.plot(np.arange(len(X)), X, label="Input", color="C0")
    plt.scatter(np.arange(len(X), len(X)+len(y)), y,label="Truth", color="C0", marker="x")
    plt.plot(np.arange(len(X), len(X)+len(pred)), pred, label=name[:-5]+"Prediction", color="C1")
    plt.plot(np.arange(len(batch)), batch, label="Simulation Prediction", color="C2")
    plt.title(f"{name[:-5]} - Simulation Comparison ", size=18)
    plt.ylim((-0.1, 1.1))
    plt.xlabel("Days", size=14)
    plt.ylabel("Cumulative cases", size=14)
    plt.legend(fontsize=12)
    plt.show()

evaluate_PP_fit(best_rnn, rnn_PP)
evaluate_PP_fit(best_lstm, lstm_PP)
