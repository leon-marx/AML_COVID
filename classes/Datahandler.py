from sys import version
import torch
import numpy as np
from torch._C import device
from Simulation import World
import matplotlib.pyplot as plt
from Toydata import create_toydata
from scipy.optimize import curve_fit
import json

class DataHandler():
    def __init__(self,mode,params,device):
        '''
        Parameters:
            mode:       Which kind of data shall this instance generate? ("SIR","Simulation","Real")
            params:     Dictionary containing the specific parameters for the selected mode
            device:     Machine on which the experiment runs
        '''

        self.device = device
        self.mode = mode

        if mode == "Simulation":
            #initialize the small world network
            W = World(N = params["N"], D = params["D"], r = params["r"], d = params["d"], N_init= params["N_init"],epsilon=params["epsilon"],version = params["version"])
            self.PP_data = torch.Tensor([params["N"], params["D"], params["r"], params["d"], params["epsilon"]])

            #Generate the full time series:
            self.cumulative = torch.zeros([params["T"]]).to(self.device)
            for i in range(params["T"]):
                self.cumulative[i] = W()

        elif mode == "Real":

            #Select a linear parts of the whole time series
            if params["full"] == False:
                
                #Load the dictionary containig the information about the linear parts
                with open("./classes/Countries/wave_regions.json","r") as file:
                    wave_dict = json.load(file)

                if params["wave"] > wave_dict[params["file"].split('.')[0]]["N_waves"]:
                    raise ValueError()

                lower = wave_dict[params["file"].split('.')[0]][f"{params['wave']}"][0]
                upper = wave_dict[params["file"].split('.')[0]][f"{params['wave']}"][1]

                self.cumulative = np.loadtxt("./classes/Countries/"+params["file"],skiprows=4)[lower:upper]
            
            #Use the full time series
            else:
                self.cumulative = np.loadtxt("./classes/Countries/"+params["file"],skiprows=4)

            #Apply moving average to smoothen the time series
            if params["use_running_average"] == True:
                self.cumulative = self.get_running_average(self.cumulative,params["dt_running_average"])

            self.cumulative = torch.tensor(self.cumulative).to(device)

        elif mode == "SIR":
            data = create_toydata(T = params["T"], I0= params["I0"], R0= params["R0"], N = params["N"], beta = params["beta"],gamma = params["gamma"])
            self.cumulative = torch.tensor(data[:,1:3].sum(-1)).to(device)

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
        if return_plain: return self.cumulative.numpy(),None
        
        #Checck if the selected lenght of the slices is valid
        if self.T <= L: raise(ValueError("Selected sequence lenght exceeds lenght of time series"))

        #Get the batch
        batch = torch.zeros([L,B,1]).to(self.device)

        #get the starting points
        starting_points = np.random.randint(0,self.T - L,B)

        for i in range(B):
            batch[:,i,0] = self.cumulative[starting_points[i]:starting_points[i]+L]

        if self.mode == "Simulation":
            return batch.view(L,B,1), starting_points, self.PP_data.repeat(L * B, 1).view(L, B, 5)
        else:
            return batch.view(L,B,1), starting_points
    
    def get_per_day(self,ts,dt = 7):
        '''
        #Compute the running average of the per day infected individuals

        Parameters:
            ts: Time series containig the cumulative cases
            dt: Time span for the moving average 

        returns:
            running_averag: running_averagof the per day cases
        '''

        #Get the per day infections
        diff = ts[1:] - ts[:-1]

        #Get the running average
        mean = np.mean(diff[:dt])

        running_averag = self.get_running_average(diff,dt)
        
        return running_averag

    def get_running_average(self,ts,dt):
        #Get the running average
        mean = np.mean(ts[:dt])

        running_averag = np.zeros(len(ts)-dt)
        running_averag[0] = mean

        for i in range(1,len(ts)-dt):
            old_lower = i-1
            new_upper = i+dt-1

            mean =  mean - ts[old_lower] / dt + ts[new_upper] / dt

            running_averag[i] = mean

        return running_averag


class Sampler():
    def __init__(self,lower_lims,upper_lims,N,T,version,device):
        '''
        parameters:
            lower_lims:     Dictionary containing the limit values of the different pandemic paramters
            upper_lims:     Dictionary containing the upper values of the different pandemic paramters
            N:              Population size
            T:              Lenght of the Simulation
            version:        Version of the Simulation
            device:         Device
        '''
        self.lower_lims = lower_lims
        self.upper_lims = upper_lims
        self.N = N
        self.T = T
        self.version = version
        self.device = device
    
    def __call__(self,K,L,B):
        '''
        parameters:
            K:      Number of Simulations from differnt pandemic parameter combinations
            L:      Lengh of the slices
            B:      Number of slices per pandemic parameter combination

        returns:
            batch:                  Tensor of slices, shape of batch: [B*K, L ,1]
            pandemic_parameters:    Pandmic parameters used to generate the corresponding entry in batch
            starting_points:        Startingpositions of the differnt sliecs in teh time series
        '''

        params_simulation = {}
        params_simulation["N"] = self.N
        params_simulation["version"] = self.version
        params_simulation["T"] = self.T

        batch = torch.zeros([B*K,L]).to(self.device)
        pandemic_parameters = torch.zeros([B*K,7]).to(self.device) #order: [[D,N,r,d,N_init,T,epsilon]]
        starting_points = torch.zeros([B*K]).to(self.device)

        for i in range(K):
            #Sample the Pandemic parameters randomly from a uniform distribution
            for k in self.lower_lims.keys():
                if k == "N_init" or k == "D":
                    params_simulation[k] = int(np.random.uniform(self.lower_lims[k],self.upper_lims[k]))
                else:
                    params_simulation[k] = np.random.uniform(self.lower_lims[k],self.upper_lims[k])

            #Get the simulation
            DH = DataHandler("Simulation",params_simulation,device=self.device)

            #Sample from the Data handler
            b,sp,PP= DH(B,L) #returns batch with shape [L,B,1]
            
            batch[i * B:(i+1) * B] = b.squeeze().T

            pandemic_parameters[i * B:(i+1) * B] = torch.tensor([params_simulation["D"],params_simulation["N"],params_simulation["r"],params_simulation["d"],params_simulation["N_init"],params_simulation["T"],params_simulation["epsilon"]])[None,:]
            starting_points[i * B : (i+1) * B] = torch.tensor(sp)
            print(sp)

        #Shuffle the batch
        indices = np.random.permutation(B * K)

        batch = batch[indices].to(self.device)
        pandemic_parameters = pandemic_parameters[indices].to(self.device)
        starting_points = starting_points[indices].to(self.device)

        #reshape the batch 
        batch = batch.T.view([L,B * K,1]).to(self.device)

        return batch,pandemic_parameters,starting_points



#####################################################################################
#Example sampler
#####################################################################################
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
'''
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
'''

#####################################################################################
#Example
#####################################################################################
params_simulation = {
    "D": 8,
    "N": 1000,
    "r": 0.1,
    "d": 14,
    "N_init": 5,
    "T":30,
    "epsilon":0.1,
    "version":"V2"
} #version V2 is the one with random flipping

params_real = {
    "file":"Israel.txt",
    "wave":4,
    "full":False,
    "use_running_average":True,
    "dt_running_average":14
}

params_SIR = {
    "T":10,
    "I0":1,
    "R0":0,
    "N":100,
    "beta":1.3,
    "gamma":0.3
}
B = 2
L = 9 

'''

params_simulation["version"] = "V1"
DH = DataHandler("Simulation",params_simulation,device = "cpu")
batch2,starting_points  = DH(B,L,return_plain=True)
plt.plot(batch2,label = "V1")

params_simulation["version"] = "V2"
DH = DataHandler("Simulation",params_simulation,device = "cpu")
batch2,starting_points  = DH(B,L,return_plain=True)
plt.plot(batch2,label = "V2")

params_simulation["version"] = "V3"
DH = DataHandler("Simulation",params_simulation,device = "cpu")
batch2,starting_points  = DH(B,L,return_plain=True)
plt.plot(batch2,label = "V3")

plt.legend()
plt.show()


'''