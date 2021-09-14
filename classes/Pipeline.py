import torch
from Datahandler import DataHandler
from Simulation import World
from matplotlib import pyplot as plt
import numpy as np
from Datahandler import Sampler

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
params_real = {
    "file":"Israel.txt",
    "wave":4,
    "full":False,
    "use_running_average":True,
    "dt_running_average":14
}

dh = DataHandler(mode='Real',device=device,params=params_real)
ts = dh.cumulative


fig, ax = plt.subplots(1)
fig.suptitle('Series')
x = np.linspace(1, len(ts), num=len(ts))
ax.plot(x, ts, label='Cumulative timeseries real')
plt.savefig('plots/ts_cumulative.jpg')
plt.close()
'''

lower_lims = {
    "D":1,
    "r":0.0,
    "d":7,
    "N_init":0,
    "epsilon":0.0
}

upper_lims = {
    "D":5,
    "r":0.5,
    "d":21,
    "N_init":20,
    "epsilon":0.5
}
L = 10
K = 3
B = 3
T = 50

S = Sampler(lower_lims,upper_lims,1000,T,"V3","cpu")

batch,pandemic_parameters,starting_points = S(K,L,B)
batch = batch.detach().view(L,K*B).numpy().T
starting_points = starting_points.detach().numpy()

#plot the slices
x = np.arange(0,L)
for i in range(K * B):
    plt.plot(x + starting_points[i],batch[i],color = "b")
    plt.xlim([0,T])

plt.legend()
plt.show()


