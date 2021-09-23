from sys import version
import torch
import numpy as np
import pandas as pd
from torch._C import device
from Simulation import World
import matplotlib.pyplot as plt
from Toydata import create_toydata
from ast import literal_eval
import json
import ast


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
            smooth_check = False
            for i in range(params["T"]):
                #Stp if the desired lengh is reached, to handle the extra loop in case of smooth transition
                if self.cumulative[-1] != 0:
                    break

                #Change the Pandemic parameters
                if i == params["T_change_D"]:
                    #Smooth the transition of the degree, by decreasing it step by step
                    if params["Smooth_transition"] == 1:
                        range_upper_lim = params["D"]-params["D_new"]
                        if range_upper_lim == 0:
                            self.cumulative[i] = W(change_now=True,D_new = params["D"],r_new = params["r_new"])
                        for j in range(0,range_upper_lim):
                            if self.cumulative[-1] != 0:
                                break
                            # print(i+j-1)
                            self.cumulative[i+j] = W(change_now=True,D_new = params["D"] - (j+1),r_new = params["r_new"])
                            smooth_check = True
                    #Hard change of D -> D_new
                    else:
                        self.cumulative[i] = W(change_now=True,D_new=params["D_new"],r_new = params["r_new"])

                else:
                    if smooth_check:
                        self.cumulative[i+params["D"]-params["D_new"]-1] = W()
                    else:
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
            Timeseries = create_toydata(T = params["T"], I0 = params["I0"], R0 = params["R0"], N = params["N"], beta = params["beta"], gamma = params["gamma"]).T
            self.cumulative = torch.tensor((Timeseries[1]+Timeseries[2]) / params["N"]).to(device)

            #Normalize with the final value of the real data
            self.cumulative /= self.cumulative[-1]


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
            batch:              B consecutive sequences of length L of the stored time series (default) as tensor of shape [L,B,1]7
                                self.cumulative in shape [len(self.cumulative)] if return_plain == True, as numpy array
            starting_points:    Starting pints of the slices
        '''

        #return the full time series
        if return_plain: return self.cumulative, None #.cpu().numpy(), None 
        
        #Check if the selected length of the slices is valid
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
    
    def __call__(self,K,L,B,mode="random"):
        '''
        parameters:
            K:      Number of Simulations from different pandemic parameter combinations
            L:      Length of the slices
            B:      Number of slices per pandemic parameter combination
            mode:   if PP combinations are chosen randomly or based on optimization on the real curves ("random", "optimized")

        returns:
            batch:                  Tensor of slices, shape of batch: [B*K, L ,1]
            pandemic_parameters:    Pandemic parameters used to generate the corresponding entry in batch (N, D, r, d, epsilon)
            starting_points:        Startingpositions of the different slices in teh time series
        '''

        print('-'*40)
        print('Started sampling')

        params_simulation = {}
        params_simulation["N"] = self.N
        params_simulation["version"] = self.version
        params_simulation["T"] = self.T

        if mode == "random":
            batch = torch.zeros([B*K,L]).to(self.device)
            pandemic_parameters = torch.zeros([B*K,9]).to(self.device) #order: (N, D, r, d, epsilon)
            starting_points = torch.zeros([B*K]).to(self.device)

            for i in range(K):
                #smooth transition or not
                params_simulation["Smooth_transition"] = torch.randint(low=0,high=2,size = [1]).item()

                #Sample the Pandemic parameters randomly from a uniform distribution
                for k in self.lower_lims.keys():
                    #integer variables
                    if k == "N_init" or k == "D" or k == "D_new" or k == "T_change_D": 
                        params_simulation[k] = int(np.random.uniform(self.lower_lims[k],self.upper_lims[k]))
                    #float variables
                    else:
                        params_simulation[k] = np.random.uniform(self.lower_lims[k],self.upper_lims[k])
                params_simulation["D_new"] = min(params_simulation["D"], params_simulation["D_new"])
                params_simulation["r_new"] = min(params_simulation["r"], params_simulation["r_new"])

                #Get the simulation
                DH = DataHandler("Simulation",params_simulation,device=self.device)

                #Sample from the Data handler
                b,_ = DH(B = None,L = None,return_plain=True)
                b = b.to("cpu")
                plt.plot(b)
                # plt.show()
                b,sp,PP= DH(B,L) #returns batch with shape [L,B,1]
                
                batch[i * B:(i+1) * B] = b.squeeze().T

                pandemic_parameters[i * B:(i+1) * B] = torch.tensor([params_simulation["N"], params_simulation["D"],params_simulation["r"],params_simulation["d"],params_simulation["epsilon"],params_simulation["D_new"],params_simulation["r_new"],params_simulation["T_change_D"],params_simulation["Smooth_transition"]])[None,:]
                starting_points[i * B : (i+1) * B] = torch.tensor(sp)
                # print(sp)

            #Shuffle the batch
            indices = np.random.permutation(B * K)

            batch = batch[indices].to(self.device)
            pandemic_parameters = pandemic_parameters[indices].repeat(L, 1, 1).to(self.device)
            starting_points = starting_points[indices].to(self.device)

            #reshape the batch 
            batch = batch.T.view([L,B * K,1]).to(self.device)
        
        elif mode == "optimized":

            #TODO include N (population size) and country 

            #Read optimized pp from file
            with open("./config.json", "r") as f: 
                config = json.load(f)
            df_pp = pd.read_csv(config["path_optimized_pp"])
            pp = df_pp['0']

            assert K <= len(pp) # number of different pp combinations
            batch = torch.zeros([B*K,L]).to(self.device)
            pandemic_parameters = torch.zeros([B*K,5]).to(self.device) #order: (N, D, r, d, epsilon)
            starting_points = torch.zeros([B*K]).to(self.device)

            #Get the simulation for all combinations
            i = 0
            for i in range(K):
                # Instantiate Data handler 
                pp_combination = ast.literal_eval(pp[i]) #cast i string(dict) -> dict
                pp_combination["N"] = params_simulation["N"]
                pp_combination["T"] = params_simulation["T"] 
                pp_combination["version"] = params_simulation["version"]

                DH = DataHandler("Simulation", pp_combination, device=self.device)

                #Sample from the Data handler
                b,sp,PP= DH(B,L) #returns batch with shape [L,B,1]
                batch[i * B:(i+1) * B] = b.squeeze().T
                pandemic_parameters[i * B:(i+1) * B] = torch.tensor([pp_combination["N"], pp_combination["D"],pp_combination["r"],pp_combination["d"],pp_combination["epsilon"]])[None,:]
                starting_points[i * B : (i+1) * B] = torch.tensor(sp)
                i+=1
        
        return batch,pandemic_parameters,starting_points, 

    def __plot_subset____(self, L, K, B, T, starting_points, batch, max_num_plots=25, path="./plots/sampler.jpg"):
        #plot the slices
        x = np.arange(0,L)
        num = min(K*B,max_num_plots,25)
        plt.figure(figsize=(12, 12))
        plt.suptitle('Samples from simulation')
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.plot(x, batch[i].cpu().numpy(), color = "b")
            plt.xlim([0,T])
            plt.legend()
        plt.savefig(path)

    def __save_to_file__(self, N, K, B, L, T, version, mode, batch, pp, path="./sampled_data.csv"):
        all_simulations = list()
        batch = batch.cpu().numpy()
        pp = pp.cpu().numpy()
        for i in range (K*B):
            simulation_instance = dict()
            simulation_instance['N'] = str(pp[0][i][0]) #L,B*K,PP in order (N, D, r, d, epsilon)
            simulation_instance['D'] = pp[0][i][1]
            simulation_instance['r'] = pp[0][i][2]
            simulation_instance['d'] = pp[0][i][3]
            simulation_instance['epsilon'] = pp[0][i][4]
            simulation_instance['K'] = K
            simulation_instance['B'] = B
            simulation_instance['L'] = L
            simulation_instance['T'] = T

            simulation_instance["D_new"] = pp[0][i][5]
            simulation_instance["r_new"] = pp[0][i][6]
            simulation_instance["Smooth_transition"] = pp[0][i][8]
            simulation_instance["T_change_D"] = pp[0][i][7]
            
            
            simulation_instance['version'] = version
            simulation_instance['mode'] = mode 
            simulation_instance['timeseries'] = list(batch[i])
            all_simulations.append(simulation_instance)

        all_simulations = pd.DataFrame(all_simulations)
        all_simulations.to_csv(path)

        #with open(path, 'w') as f:
        #    json.dump(all_simulations,f)
        #all = pd.DataFrame([(PP[i],data[i]) for i in range(len(data))], columns=['pp','timeseries'])
        #all.to_csv(path)

    def __load_from_file__(self, L, path="./sampled_data.csv"):
        all = pd.read_csv(path)
        data, PP = torch.tensor([literal_eval(ts) for ts in all['timeseries']]), torch.stack((torch.tensor(all['N']),torch.tensor(all['D']),torch.tensor(all['r']),torch.tensor(all['d']),torch.tensor(all['epsilon']))).T #(N, D, r, d, epsilon)
        return data, PP.repeat(L, 1, 1) 

if __name__ == "__main__":

#####################################################################################
#Example Sampler
#####################################################################################
    lower_lims = {
        "D":1,
        "D_new":1,
        "r_new":0.001,
        "T_change_D":0,
        "r":0.001,
        "d":7,
        "N_init":1,
        "epsilon":0.0
    }

    upper_lims = {
        "D":5,
        "D_new":5,
        "r_new":0.5,
        "T_change_D":50,
        "r":0.5,
        "d":21,
        "N_init":20,
        "epsilon":0.5
    }

    N = 100 #10000
    L = 20 #10
    K = 10 #1000
    B = 1 #20
    T = 50
    version = "V2"
    mode = "random"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = f"./sampled_data_mode_{mode}_version_{version}_N-{N}_K-{K}.csv"

    #Random Sampler 
    #S = Sampler(lower_lims,upper_lims,1000,T,"V3","cpu")
    #batch,pandemic_parameters,starting_points = S(K,L,B)
    #batch = batch.detach().view(L,K*B).numpy().T
    #starting_points = starting_points.detach().numpy()

    #Optimized Sampler 
    S = Sampler(lower_lims=lower_lims, upper_lims=upper_lims, N=N, T=T, version=version, device=device)
    batch,pandemic_parameters,starting_points = S(K=K,L=L,B=B,mode=mode)
    batch = batch.detach().view(L,K*B).T
    starting_points = starting_points.detach()
    S.__plot_subset____(L=L, K=K, B=B, T=T, starting_points=starting_points, batch=batch, path="./plots/samplertest.png")
    S.__save_to_file__(N=N, K=K, B=B, L=L, T=T, version=version, mode=mode, batch=batch,pp=pandemic_parameters,path=path)
    batch_recov, pp_recov = S.__load_from_file__(L=L, path=path)
    # print(batch_recov)
    # print(pp_recov)


def compare_hard_smooth_transition():
    params_simulation = {
        "D": 8,
        "D_new":3,
        "r_new":0.1,
        "T_change_D":10,
        "N": 1000,
        "r": 0.1,
        "d": 14,
        "N_init": 5,
        "T":30,
        "epsilon":0.1,
        "version":"V2",
        "Smooth_transition":1
    }

    plt.figure(figsize = (30,15))

    fs = 30

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases []",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlim([-1, params_simulation["T"]])
    plt.ylim([0,1.2])

    params_simulation["version"] = "V2"
    params_simulation["Smooth_transition"] = 0
    DH = DataHandler("Simulation",params_simulation,device = "cpu")
    batch2,starting_points  = DH(B = None,L= None,return_plain=True)
    plt.plot(batch2,label = f"V2, hard transition D = {params_simulation['D']} -> D = {params_simulation['D_new']}",linewidth = 6)

    params_simulation["Smooth_transition"] = 1
    DH = DataHandler("Simulation",params_simulation,device = "cpu")
    batch2,starting_points  = DH(B = None,L= None,return_plain=True)
    plt.plot(batch2,label = f"V2, soft transition D = {params_simulation['D']} -> D = {params_simulation['D_new']}",linewidth = 6)

    params_simulation["Smooth_transition"] = 1
    params_simulation["T_change_D"] = 10000
    DH = DataHandler("Simulation",params_simulation,device = "cpu")
    batch2,starting_points  = DH(B = None,L= None,return_plain=True)
    plt.plot(batch2,label = f"V2, constant D = {params_simulation['D']}",linewidth = 6)

    plt.legend(fontsize = fs)

    plt.savefig(f"./compare_hard_smooth_transition.jpg")

    plt.show()
    
#####################################################################################
#Example
#####################################################################################
params_simulation = {
    "D": 8,
    "D_new":3,
    "r_new":0.1,
    "T_change_D":10,
    "N": 1000,
    "r": 0.1,
    "d": 14,
    "N_init": 5,
    "T":30,
    "epsilon":0.1,
    "version":"V2",
    "Smooth_transition":1,
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

#params_simulation["version"] = "V1"
#DH = DataHandler("Simulation",params_simulation,device = "cpu")
#batch2,starting_points  = DH(B,L,return_plain=True)
#plt.plot(batch2,label = "V1")
"""
params_simulation["version"] = "V2"
params_simulation["Smooth_transition"] = 0
DH = DataHandler("Simulation",params_simulation,device = "cpu")
batch2,starting_points  = DH(B,L,return_plain=True)
plt.plot(batch2,label = "V2, hard transition")

params_simulation["Smooth_transition"] = 1
DH = DataHandler("Simulation",params_simulation,device = "cpu")
batch2,starting_points  = DH(B,L,return_plain=True)
plt.plot(batch2,label = "V2, smooth transition")"""

#params_simulation["version"] = "V3"
#DH = DataHandler("Simulation",params_simulation,device = "cpu")
#batch2,starting_points  = DH(B,L,return_plain=True)
#plt.plot(batch2,label = "V3")

"""plt.legend()
plt.show()"""
