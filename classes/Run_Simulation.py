from Simulation import World
from Toydata import create_toydata
import matplotlib
from matplotlib import pyplot as plt
import torch
import numpy as np

# Parameters for the simulation 
N = 100 # population size (number of individuals) #= 100  
D = 8 # degree, number of contact persons
r = 0.2 # rate of passing the infection
d = 14 # duration of the infection
N_init = 5 #number of persons infected at t_0; = 1 

# Parameters for the SIR model 
gamma = 1/d # death / recovery rate per day = 0.3
beta = r * D / N # number of new infections for one first infected person per day = 1.3 
R0 = 0 # initial number of recovered people
I0 = 5 # corresponds to N_init 

T = 200 #number of days

# Run the simulation 
W = World(N = N, D = D, r = r,d = d, N_init = N_init)
simulation = []
for i in range(T):
    if i % 1 == 0: W.plotter(f"plots/Step_{i}.jpg")
    simulation.append(float(W())*N)
print(simulation)

# Create toy data
sir_model = create_toydata(T, I0, R0, N, beta, gamma)
print(sir_model)

# Plot the two series 
fig, ax = plt.subplots(1)
fig.suptitle('SIR vs Simulation')
x = np.linspace(1, T, num=T)
ax.plot(x, simulation, label='Cumulative simulation')
ax.plot(x, sir_model[:,1]+sir_model[:,2], label='Cumulative SIR')
plt.legend()
plt.savefig('plots/sir_vs_sim.jpg')
plt.close()


# TODO 
# 1. calibrate simulation on real data  
# 2. run calibration with parameters from SIR model 
# 3. part for toy data from SIR model with same parameters as simulation 
# compare time series with loss function 