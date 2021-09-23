# Imports
import torch
import Datahandler
import Dataset

# Parameters
lower_lims = {
    "D": 1,
    "D_new": 1,
    "r": 0.0,
    "r_new": 0.0,
    "d": 3,
    "N_init": 1,
    "epsilon": 0.0,
    "T_change_D": 0
}
upper_lims = {
    "D": 10,
    "D_new": 10,
    "r": 0.5,
    "r_new": 0.5,
    "d": 21,
    "N_init": 5,
    "epsilon": 0.7,
    "T_change_D": 50
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
train_ratio = 0.7  # how much of the data is used as the training set

# Generating Data and saving it
mysampler = Datahandler.Sampler(
    lower_lims=lower_lims,
    upper_lims=upper_lims,
    N=N,
    T=T,
    version=version,
    device=device
)
# batch, pandemic_parameters, starting_points = mysampler(K, L, B)
# torch.save(batch, "data_path.pt")
# torch.save(pandemic_parameters[:,:,:5], "PP_path.pt")

# Loading data divided into train and test set
# batch = torch.load("data_path.pt")
# pandemic_parameters = torch.load("PP_path.pt")
# print(pandemic_parameters.shape)
batch_length = 250
train_inds, test_inds = torch.utils.data.random_split(
    torch.arange(batch_length), 
    [int(batch_length*train_ratio), batch_length - int(batch_length*train_ratio)], 
    generator=torch.Generator().manual_seed(17))

# Using the Dataset class
train_data = Dataset.Dataset("data_path.pt", "PP_path.pt", train_inds, backtime=backtime, foretime=foretime)
test_data = Dataset.Dataset("data_path.pt", "PP_path.pt", test_inds, backtime=backtime, foretime=foretime)
