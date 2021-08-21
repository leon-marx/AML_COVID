import torch
import numpy as np
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

        if mode == "Simulation":
            #initialize the small world network
            W = World(N = params["N"], D = params["D"], r = params["r"], d = params["d"], N_init= params["N_init"])

            #Generate the full time series:
            self.cumulative = torch.zeros([params["T"]]).to(self.device)
            for i in range(params["T"]):
                self.cumulative[i] = W()

        elif mode == "Real":

            #Select a linear parts of the whole time series
            if params["full"] == False:
                
                #Load the dictionary containig the information about the linear parts
                with open("./Countries/wave_regions.json","r") as file:
                    wave_dict = json.load(file)

                if params["wave"] > wave_dict[params["file"].split('.')[0]]["N_waves"]:
                    raise ValueError()

                lower = wave_dict[params["file"].split('.')[0]][f"{params['wave']}"][0]
                upper = wave_dict[params["file"].split('.')[0]][f"{params['wave']}"][1]

                self.cumulative = np.loadtxt("./Countries/"+params["file"],skiprows=4)[lower:upper]
            
            #Use the full time series
            else:
                self.cumulative = np.loadtxt("./Countries/"+params["file"],skiprows=4)

            #Apply moving average to smoothen the time series
            if params["use_running_average"] == True:
                self.cumulative = self.get_running_average(self.cumulative,params["dt_running_average"])

            self.cumulative = torch.tensor(self.cumulative).to(device)

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
        if return_plain: return self.cumulative.numpy(),None
        
        #Checck if the selected lenght of the slices is valid
        if self.T <= L: raise(ValueError("Selected sequence lenght exceeds lenght of time series"))

        #Get the batch
        batch = torch.zeros([L,B,1]).to(self.device)

        #get the starting points
        starting_points = np.random.randint(0,self.T - L,B)

        for i in range(B):
            batch[:,i,0] = self.cumulative[starting_points[i]:starting_points[i]+L]

        return batch.view(L,B,1),starting_points

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
    "nu":0.001,
    "beta":1.3,
    "gamma":0.3,
    "mu":0.0001
}

DH = DataHandler("Real",params_real,device = "cpu")

B = 2
L = 9 

<<<<<<< HEAD
batch,starting_points  = DH(B,L,return_plain=True)
per_day = DH.get_per_day(batch)

plt.subplot(2,1,1)
plt.plot(batch)

plt.subplot(2,1,2)
plt.plot(per_day)


=======
batch,starting_points  = DH(B,L)
print(batch.shape)
plt.plot(DH(B,L,return_plain=True))
for i in range(B):
    plt.plot(np.arange(starting_points[i],starting_points[i]+L),batch[:,i].detach().numpy(),ls = "",marker = "+")
>>>>>>> ebe0770031a588c1aa342272f08084366b8fe795
plt.show()
'''

