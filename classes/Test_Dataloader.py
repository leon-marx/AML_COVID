# Imports
import torch
import Datahandler
import Dataset

# Parameters
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
T = 50  # length of the simulation   AS BIG AS POSSIBLE WITHOUT FLAT
version = "V3"
device = "cuda" if torch.cuda.is_available() else "cpu"
K = 50  # number of simulations sampled from the PP ranges   AS BIG AS POSSIBLE
L = T - 5  # length of the slices
B = 5  # number of slices per simulation
backtime = 20  # number of days the network gets to see before prediction
foretime = 3  # number of days to predict for long predictions

# Generating Data and saving it
mysampler = Datahandler.Sampler(
    lower_lims=lower_lims,
    upper_lims=upper_lims,
    N=N,
    T=T,
    version=version,
    device=device
)
batch, pandemic_parameters, starting_points = mysampler(K, L, B)
torch.save(batch, "data_path")
torch.save(pandemic_parameters, "PP_path")

# Loading data divided into train and test set
