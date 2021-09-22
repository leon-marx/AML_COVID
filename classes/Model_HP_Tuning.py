# Package Imports
import torch

# Own Imports
import Datahandler
import LSTM_Model

# Fixed Parameters
input_size = 1
output_size = 1
lower_lims = {
    "D": 1,
    "r": 0.0,
    "d": 3,
    "N_init": 1,
    "epsilon": 0.0
}
upper_lims = {
    "D": 10,
    "r": 0.5,
    "d": 21,
    "N_init": 5,
    "epsilon": 0.7
}
N = 1000  # population size
T = 50  # length of the simulation
version = "V3"
device = "cuda" if torch.cuda.is_available() else "cpu"
K = 10  # number of simulations sampled from the PP ranges
L = 20  # length of the slices
B = 10  # number of slices per simulation
test_batch_size = 4
foretime = 3
backtime = 20
n_epochs = 1000
LOG_FOLDER = "Tuning_Logs"

# Parameters to Tune
hidden_size_list = [128, 256, 512]
num_layers_list = [1, 2, 4]
dropout_list = [0.0, 0.5]
learning_rate_list = [0.00001, 0.0001, 0.001, 0.01]

# Sampler Initialization
mysampler = Datahandler.Sampler(lower_lims=lower_lims,
                                upper_lims=upper_lims,
                                N=N,
                                T=T,
                                version=version,
                                device=device)

# Data Generation
print("Generating Data")
batch, pandemic_parameters, starting_points = mysampler(K, L, B)
training_data = batch[:,:-test_batch_size,...]
test_data = batch[:,-test_batch_size:,...]
training_PP = pandemic_parameters[:,:-test_batch_size,...]
test_PP = pandemic_parameters[:,-test_batch_size:,...]

def tune_lstm(hidden_size, num_layers, dropout, learning_rate):
    # Model
    print("Initializing Model with:")
    print(f"    hidden_size: {hidden_size}")
    print(f"    num_layers: {num_layers}")
    print(f"    dropout: {dropout}")
    print(f"    learning_rate: {learning_rate}")
    mylstm = LSTM_Model.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            foretime=foretime,
                            backtime=backtime)
    # LSTM Training
    print("Training LSTM")
    mylstm.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=mylstm.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(n_epochs):
        mylstm.train_model(training_data=training_data,training_PP=training_PP, loss_fn=loss_fn, optimizer=optimizer)
        if epoch % 100 == 0:
            test_loss = mylstm.test_model(test_data=test_data, test_PP=test_PP, loss_fn=loss_fn)
            losses.append(test_loss)
    with open(f"{LOG_FOLDER}/LSTM-hidden_size_{hidden_size}-num_layers_{num_layers}-dropout{dropout}-learning_rate_{learning_rate}.txt", "w") as file:
        for loss in losses:
            file.write(str(loss) + "\n")
    print("")
    print("")
    print("")

print("Running Loop")
for hidden_size in hidden_size_list:
    for num_layers in num_layers_list:
        for learning_rate in learning_rate_list:
            for dropout in dropout_list:
                tune_lstm(n_epochs, hidden_size, num_layers, dropout, learning_rate)
