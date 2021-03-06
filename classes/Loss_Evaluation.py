import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import LSTM_Model
import RNN_Model
import Dataset
import Datahandler

data_version = 2
directory = "C:/Users/gooog/Desktop/AML/final_project/repository/AML_COVID/data/Evaluation/"
if data_version == 1:
    batch_length = 20
else:
    batch_length = 20000
backtime = 20
foretime = 5
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

def get_losses(old_only=False):
    names = []
    losses = []
    for filename in os.listdir(directory):
        if "comparison" in filename:
            continue
        if old_only:
            if "foretime" in filename:
                continue
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
            break
        names.append(filename)
        losses.append(np.mean(loss))
    return names, losses

def plot_predictions(filename, many=False, length=foretime):
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
    def new_my_collate(batch):
        x = torch.zeros(size=(backtime, len(batch), 1))
        pp = torch.zeros(size=(backtime, len(batch), 5))
        y = torch.zeros(size=(length, len(batch), 1))
        for i, item in enumerate(batch):
            x[:, i, :] = item[0]
            pp[:, i, :] = item[1]
            y[:, i, :] = item[2]
        return x, pp, y
    new_training_data = Dataset.Dataset(DATA_PATH, PP_PATH, train_inds, backtime=backtime, foretime=length)
    new_test_data = Dataset.Dataset(DATA_PATH, PP_PATH, test_inds, backtime=backtime, foretime=length)
    new_training_dataloader = torch.utils.data.DataLoader(new_training_data, batch_size=batch_size, shuffle=True, collate_fn=new_my_collate)
    new_test_dataloader = torch.utils.data.DataLoader(new_test_data, batch_size=batch_size, shuffle=True, collate_fn=new_my_collate)
    Xs = 0
    PPs = 0
    ys = 0
    plt.figure(figsize=(12, 8))
    for Xt, PPt, yt in new_test_dataloader:
        Xs = Xt
        PPs = PPt
        ys = yt
        break
    for i in range(9):
        plt.subplot(3, 3, i+1)
        X = Xs[:, i, :].view(-1, 1, 1)
        PP = PPs[:, i, :].view(-1, 1, 5)
        y = ys[:, i, :].view(-1, 1, 1)
        model.to(device)
        pred = model.predict_long(X, PP, 20)
        X = X.view(-1)
        y = y.view(-1)
        pred = pred.view(-1).detach().numpy()
        plt.plot(np.arange(len(X)), X, label="Input", color="C0")
        plt.scatter(np.arange(len(X), len(X)+len(y)), y,label="Truth", color="C0", marker="x")
        plt.plot(np.arange(len(X), len(X)+len(pred)), pred, label="Prediction", color="C1")
        plt.ylim((-0.1, 1.1))
        plt.xlabel("Days", size=14)
        plt.ylabel("Cumulative cases", size=14)
        plt.legend(fontsize=12)
    plt.suptitle(filename.replace("_", " ").replace("-", ", "), size=18)
    if many:
        plt.savefig(directory + "plots/many/prediction_" + f"_{filename}.png")
        plt.close()
    else:
        plt.savefig(directory + "plots/prediction_" + name + f"_{foretime}_png")
        plt.close()

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

def get_best(names, losses):
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
            else:
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
            else :
                if losses[i] < best_lstm_00001_loss:
                    best_lstm_00001 = name
                    best_lstm_00001_loss = losses[i]
    return best_rnn_01, best_rnn_01_loss, best_rnn_001, best_rnn_001_loss, best_rnn_0001, best_rnn_0001_loss, best_rnn_00001, best_rnn_00001_loss, best_lstm_01, best_lstm_01_loss, best_lstm_001, best_lstm_001_loss, best_lstm_0001, best_lstm_0001_loss, best_lstm_00001, best_lstm_00001_loss

def run_comparison(names, losses):
    best_relu, best_relu_loss, best_tanh, best_tanh_loss = compare_nonlinearity(names, losses)
    best_drop, best_drop_loss, best_nodrop, best_nodrop_loss = compare_dropout(names, losses)
    best_rnn_1, best_rnn_1_loss, best_rnn_2, best_rnn_2_loss, best_lstm_1, best_lstm_1_loss, best_lstm_2, best_lstm_2_loss = compare_layers(names, losses)
    best_rnn_128, best_rnn_128_loss, best_rnn_256, best_rnn_256_loss, best_rnn_512, best_rnn_512_loss, best_lstm_128, best_lstm_128_loss, best_lstm_256, best_lstm_256_loss, best_lstm_512, best_lstm_512_loss = compare_hidden(names, losses)
    best_rnn_01, best_rnn_01_loss, best_rnn_001, best_rnn_001_loss, best_rnn_0001, best_rnn_0001_loss, best_rnn_00001, best_rnn_00001_loss, best_lstm_01, best_lstm_01_loss, best_lstm_001, best_lstm_001_loss, best_lstm_0001, best_lstm_0001_loss, best_lstm_00001, best_lstm_00001_loss = compare_lrs(names, losses)

    with open(directory + "comparisons/nonlinearity.txt", "w") as f:
        f.write(f"ReLU & Tanh \n")
        f.write(f"\hline \n")
        f.write(f"{np.round(best_relu_loss, 5)} & {np.round(best_tanh_loss, 5)} \n")
        f.write("")

    with open(directory + "comparisons/dropout.txt", "w") as f:
        f.write(f"Dropout & No Dropout \n")
        f.write(f"\hline \n")
        f.write(f"{np.round(best_drop_loss, 5)} & {np.round(best_nodrop_loss, 5)} \n")
        f.write("")

    with open(directory + "comparisons/num_layers.txt", "w") as f:
        f.write(f"Model & 1 Layer & 2 Layers \n")
        f.write(f"\hline \n")
        f.write(f"RNN & {np.round(best_rnn_1_loss, 5)} & {np.round(best_rnn_2_loss, 5)} \n")
        f.write(f"LSTM & {np.round(best_lstm_1_loss, 5)} & {np.round(best_lstm_2_loss, 5)} \n")
        f.write("")

    with open(directory + "comparisons/hidden_size.txt", "w") as f:
        f.write(f"Model & 128 Neurons & 256 Neurons & 512 Neurons \n")
        f.write(f"\hline \n")
        f.write(f"RNN & {np.round(best_rnn_128_loss, 5)} & {np.round(best_rnn_256_loss, 5)} & {np.round(best_rnn_512_loss, 5)} \n")
        f.write(f"LSTM & {np.round(best_lstm_128_loss, 5)} & {np.round(best_lstm_256_loss, 5)} & {np.round(best_lstm_512_loss, 5)} \n")
        f.write("")

    with open(directory + "comparisons/learning_rate.txt", "w") as f:
        f.write(f"Model & 0.01 & 0.001 & 0.0001 & 0.00001 \n")
        f.write(f"\hline \n")
        f.write(f"RNN & {np.round(best_rnn_01_loss, 5)} & {np.round(best_rnn_001_loss, 5)} & {np.round(best_rnn_0001_loss, 5)} & {np.round(best_rnn_00001_loss, 5)} \n")
        f.write(f"LSTM & {np.round(best_lstm_01_loss, 5)} & {np.round(best_lstm_001_loss, 5)} & {np.round(best_lstm_0001_loss, 5)} & {np.round(best_lstm_00001_loss, 5)} \n")
        f.write("")

def plot_loss_decay(names, losses):
    if rnn_check:
        rnn_train_loss = np.loadtxt(directory + best_rnn[:3] + best_rnn[9:-3] + "_train.txt")
    else:
        rnn_train_loss = np.loadtxt(directory + best_rnn[:3] + best_rnn[9:] + "_train.txt")
    if lstm_check:
        lstm_train_loss = np.loadtxt(directory + best_lstm[:4] + best_lstm[10:-3] + "_train.txt")
    else:
        lstm_train_loss = np.loadtxt(directory + best_lstm[:4] + best_lstm[10:] + "_train.txt")

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(rnn_train_loss)), rnn_train_loss, label="Train Loss", color="C0")
    plt.title(best_rnn.replace("_", " ").replace("-", ", "), size=18)
    plt.xlabel("Epoch", size=14)
    plt.ylabel("Training Loss", size=14)
    plt.legend(fontsize=12)
    plt.savefig(directory + "plots/prediction_RNN_png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(lstm_train_loss)), lstm_train_loss, label="Train Loss", color="C0")
    plt.title(best_lstm.replace("_", " ").replace("-", ", "), size=18)
    plt.xlabel("Epoch", size=14)
    plt.ylabel("Training Loss", size=14)
    plt.legend(fontsize=12)
    plt.savefig(directory + "plots/prediction_LSTM_png")
    plt.close()

def run_PP_fit(filename):
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
    model.to(device)
    params_real = {"file": "Israel.txt", "wave": 4, "full": False, "use_running_average": True, "dt_running_average": 14}
    DH = Datahandler.DataHandler("Real", params_real, device=device)
    fit_data, starting_points = DH(B=None, L=None, return_plain=True)

    PP_min = torch.Tensor([1, 1, 0.01, 4, 0.01])
    PP_max = torch.Tensor([10, 10, 0.15, 7, 0.1])
    PP_step = torch.Tensor([2, 1, 0.01, 1, 0.02])

    fitting_PP, fitting_loss = model.apply_PP_fit(fit_data.view(fit_data.shape[0], 1, 1).type(torch.float32), PP_min, PP_max, PP_step, loss_fn)
    torch.save(fitting_PP, directory + "NN_PP_fit/fitted_PPs_" + name)
    torch.save(fitting_loss, directory + "NN_PP_fit/fitting_loss_" + name)

names, losses = get_losses(old_only=False)
run_comparison(names, losses)
# for name in names:
#     plot_predictions(name, many=True, length=20)
# best_rnn, best_rnn_loss, best_lstm, best_lstm_loss = get_best(names, losses)
# best_rnn = input("Best RNN: ")
# best_lstm = input("Best LSTM: ")
best_rnn = "RNN_Model-hidden_size_128-num_layers_1-nonlinearity_tanh-learning_rate_1e-05"
best_lstm = "LSTM_Model-hidden_size_128-num_layers_1-dropout_0.0-learning_rate_0.0001"
rnn_check = False
if "foretime" in best_rnn:
    print("RNN GOOD")
    rnn_check = True
else:
    print("RNN BAD")
lstm_check = False
if "foretime" in best_lstm:
    print("LSTM GOOD")
    lstm_check = True
else:
    print("LSTM BAD")
plot_predictions(best_rnn)
plot_predictions(best_lstm)
plot_loss_decay(names, losses)
run_PP_fit(best_rnn)
run_PP_fit(best_lstm)
