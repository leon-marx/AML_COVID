import torch
import numpy as np
from Simulation import World
import matplotlib.pyplot as plt
from Toydata import create_toydata

class DataHandler():
    def __init__(self,mode,params,device):
        '''
        Parameters:
            mode:       Which kind of data shall this instance generate? ("SIR","Simulation","Real")
            params:     Dictionary containing the specific parameters for the selected mode
            device:     Machine on which the experiment runs
        '''

        self.device = device

        if mode == "Simulation":
            #initialize the small world network
            W = World(N = params["N"], D = params["D"], r = params["r"], d = params["d"], N_init= params["N_init"])

            #Generate the full time series:
            self.cumulative = torch.zeros([params["T"]]).to(self.device)
            for i in range(params["T"]):
                self.cumulative[i] = W()

        elif mode == "Real":
            self.cumulative = torch.tensor(np.loadtxt("./Countries/"+params["file"],skiprows=4)).to(device)

        elif mode == "SIR":
            data = create_toydata(T = params["T"], I0= params["I0"], R0= params["R0"], N = params["N"], nu = params["nu"], beta = params["beta"],gamma = params["gamma"], mu = params["mu"])
            self.cumulative = torch.tensor(data[:,2]).to(device)

        else:
            raise(NotImplementedError("Select valid data source!"))

        self.T = len(self.cumulative)

    def __call__(self,B,L,return_plain = False):
        '''
        Parameters:
            B: Batch size
            L: lenght of the sequence
            return_plain: return the whole time series, default = False

        Returns:
            batch:              B consecutive sequences of lenght L of the stored time series (default) as tensor of shape [L,B,1]7
                                self.cumulative in shape [len(self.cumulative)] if return_plain == True, as numpy array
            starting_points:    Starting pints of the slices
        '''

        #return the full time series
        if return_plain: return self.cumulative.numpy()
        
        #Checck if the selected lenght of the slices is valid
        if self.T <= L: raise(ValueError("Selected sequence lenght exceeds lenght of time series"))

        #Get the batch
        batch = torch.zeros([L,B,1]).to(self.device)

        #get the starting points
        starting_points = np.random.randint(0,self.T - L,B)

        for i in range(B):
            batch[:,i,0] = self.cumulative[starting_points[i]:starting_points[i]+L]

        return batch.view(L,B,1),starting_points


'''
params_simulation = {
    "D": 4,
    "N": 100,
    "r": 0.1,
    "d": 14,
    "N_init": 5,
    "T":50
}

params_real = {
    "file":"Germany.txt"
}

params_SIR = {
    "T":10,
    "I0":1,
    "R0":0,
    "N":100,
    "nu":0.001,
    "beta":1.3,
    "gamma":0.3,
    "mu":0.0001
}

DH = DataHandler("SIR",params_SIR,device = "cpu")
B = 2
L = 9 

batch,starting_points  = DH(B,L)

print(batch.shape)

plt.plot(DH(B,L,return_plain=True))

for i in range(B):
    plt.plot(np.arange(starting_points[i],starting_points[i]+L),batch[:,i].detach().numpy(),ls = "",marker = "+")

plt.show()
'''
