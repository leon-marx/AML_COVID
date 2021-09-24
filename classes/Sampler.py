from sys import version
import torch
import numpy as np
import pandas as pd
from torch._C import device
from Simulation import World
from Datahandler import DataHandler
import matplotlib.pyplot as plt
from Toydata import create_toydata
from ast import literal_eval
import json
import ast
import random
import os
import tqdm
import time


class Sampler():
    def __init__(self,lower_lims,upper_lims,N,T,version,device,path_to_optimized):
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
        self.path_to_optimized = path_to_optimized
    
    def __call__(self,K,L,B,mse_threshold=0.1,mode="random"):
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

        batch = torch.zeros([B*K,L]).to(self.device)
        pandemic_parameters = torch.zeros([B*K,L,5]).to(self.device) #order: (N, D, r, d, epsilon); #N, D, r, d, epsilon, D_new, r_new, T_change_D, Smooth_transition
        starting_points = torch.zeros([B*K]).to(self.device)

        # Sample random parameters 
        if mode == "random":

            for i in tqdm.tqdm(range(K)):
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
                b,sp,PP= DH(B,L) #returns batch with shape [L,B,1] 
                
                # changing D and r 
                D = torch.zeros([L,1]).to(self.device)
                r = torch.zeros([L,1]).to(self.device)
                D[0:params_simulation["T_change_D"]] = params_simulation["D"]
                D[params_simulation["T_change_D"]:] = params_simulation["D_new"]
                r[0:params_simulation["T_change_D"]] = params_simulation["r"]
                r[params_simulation["T_change_D"]:] = params_simulation["r_new"]

                # parse rest of parameters 
                torch_N = torch.tensor(params_simulation["N"]).repeat(L,1).to(self.device)
                torch_d = torch.tensor(params_simulation["d"]).repeat(L,1).to(self.device)
                torch_epsilon = torch.tensor(params_simulation["epsilon"]).repeat(L,1).to(self.device)

                # append to timeseries
                batch[i * B:(i+1) * B] = b.squeeze().T
                pandemic_parameters[i * B:(i+1) * B] = torch.cat((torch_N,D,r,torch_d,torch_epsilon),dim=1).unsqueeze(dim=0)
                starting_points[i * B : (i+1) * B] = torch.tensor(sp)

            #Shuffle the batch
            indices = np.random.permutation(B * K)
            batch = batch[indices].to(self.device)
            pandemic_parameters = pandemic_parameters[indices].to(self.device) #.repeat(L, 1, 1).to(self.device)
            starting_points = starting_points[indices].to(self.device)

            #reshape the batch 
            batch = batch.T.view([L,B * K,1]).to(self.device)
        
        # Use pp's from simulation fitted on real data
        elif mode == "optimized":

            #Get pp from different files 
            files = os.listdir(self.path_to_optimized)
            parameter_list = list()
            cost_list = list()
            unified_sim_param_list = list()
            for file in files:
                if file.endswith('.csv'):
                    parameters, cost = self.__load_from_optimized_file__(f"{self.path_to_optimized}/{file}")
                    parameter_list.append(parameters)
                    cost_list.append(cost)

            assert len(parameter_list) == len(cost_list)

            # unify lists pp_all = torch.cat((pp_all, p), dim=0)
            unified_sim_param_list = list()
            for i in range(len(parameter_list)):
                costs = cost_list[i]
                parameters = parameter_list[i]
                for c in range(len(costs)):
                    if costs[c] < mse_threshold:
                        unified_sim_param_list.append(parameters[c])

            #Shuffle torches
            random.shuffle(unified_sim_param_list)
            print(f"Found {len(unified_sim_param_list)} pp combinations with mse < {mse_threshold}")
            
            #Get the simulation for all combinations
            i = 0
            print('-'*40)
            print(f"Simulating with {K} pp combinations amd {B} batches:")
            assert K <= len(unified_sim_param_list)
            for i in range(K):
                #Track execution time
                t0 = time.perf_counter()

                # Instantiate Data handler 
                pp_sim = ast.literal_eval(unified_sim_param_list[i]) #cast i string(dict) -> dict

                # Use N,T and version given to Sampler 
                pp_sim["N"] = params_simulation["N"]
                pp_sim["T"] = params_simulation["T"] 
                pp_sim["version"] = params_simulation["version"]

                # Run simulation with pps
                DH = DataHandler("Simulation", pp_sim, device=self.device)
                b,sp,PP= DH(B,L) #returns batch with shape [L,B,1]

                # Changing D and r 
                D = torch.zeros([L,1]).to(self.device)
                r = torch.zeros([L,1]).to(self.device)
                D[0:pp_sim["T_change_D"]] = pp_sim["D"]
                D[pp_sim["T_change_D"]:] = pp_sim["D_new"]
                r[0:pp_sim["T_change_D"]] = pp_sim["r"]
                r[pp_sim["T_change_D"]:] = pp_sim["r_new"]

                # parse rest of parameters 
                torch_N = torch.tensor(pp_sim["N"]).repeat(L,1).to(self.device)
                torch_d = torch.tensor(pp_sim["d"]).repeat(L,1).to(self.device)
                torch_epsilon = torch.tensor(pp_sim["epsilon"]).repeat(L,1).to(self.device)

                # append to timeseries
                batch[i * B:(i+1) * B] = b.squeeze().T
                pandemic_parameters[i * B:(i+1) * B] = torch.cat((torch_N,D,r,torch_d,torch_epsilon),dim=1).unsqueeze(dim=0)
                starting_points[i * B : (i+1) * B] = torch.tensor(sp)

                print(f"{i+1}/{K} in {(time.perf_counter()-t0):.2f}s")

            #reshape the batch 
            batch = batch.T.view([L,B * K,1]).to(self.device)
            pandemic_parameters = pandemic_parameters.view([L,B * K,5]).to(self.device)
        
        return batch,pandemic_parameters,starting_points, 

    def __plot_simulation_subset____(self, L, K, B, T, starting_points, batch, max_num_plots=25, path="./plots/sampler.jpg"):
        
        #Reshape
        x = np.arange(0,L)
        num = min(K*B,max_num_plots,25)
        batch = batch.T.squeeze()
        
        #Plot the slices
        plt.figure(figsize=(12, 12))
        plt.suptitle('Samples from simulation')
        for i in range(num):
            plt.subplot(5, 5, i+1)
            plt.plot(x, batch[i].cpu().numpy(), color = "b")
            plt.legend()
        plt.savefig(path)

    def __save_pp_cost_to_file__(self, N, K, B, L, T, version, mode, batch, pp, path="./sampled_data.csv"):
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

            #simulation_instance["D_new"] = pp[0][i][5]
            #simulation_instance["r_new"] = pp[0][i][6]
            #simulation_instance["Smooth_transition"] = pp[0][i][8]
            #simulation_instance["T_change_D"] = pp[0][i][7]
            
            simulation_instance['version'] = version
            simulation_instance['mode'] = mode 
            simulation_instance['timeseries'] = list(batch[i])
            all_simulations.append(simulation_instance)

        all_simulations = pd.DataFrame(all_simulations)
        all_simulations.to_csv(path)

    def __save_to_file_for_training__(self, data, pandemic_parameters, version_name):
        
        path = f"{os.getcwd()}/trainingdata/{version_name}/"

        if not os.path.exists(path): 
            os.makedirs(path)

        torch.save(data.cpu(),f"./trainingdata/{version_name}/data_{version_name}.pt")
        torch.save(pandemic_parameters.cpu(),f"./trainingdata/{version_name}/pp_{version_name}.pt")


    def __load_from_sampled_file__(self, L, path="./sampled_data.csv"):
        all = pd.read_csv(path)
        data, PP = torch.tensor([literal_eval(ts) for ts in all['timeseries']]), torch.stack((torch.tensor(all['N']),torch.tensor(all['D']),torch.tensor(all['r']),torch.tensor(all['d']),torch.tensor(all['epsilon']))).T #(N, D, r, d, epsilon)
        return data, PP.repeat(L, 1, 1) 

    def __load_from_optimized_file__(self, path="./sampled_data.csv"):
        content = pd.read_csv(path)
        parameters = content['0']
        cost = content['1']
        return parameters, cost
    
    def __load_from_training_file__(self, version_name):
        data = torch.load(f"./trainingdata/{version_name}/data_{version_name}.pt")
        pp = torch.load(f"./trainingdata/{version_name}/pp_{version_name}.pt")
        return data, pp


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

    N = 2500 #10000
    L = 50 #10 datahandler?
    K = 1000 #1000
    B = 2 #20
    T = 60 # simulation?
    mse_threshold = 0.15
    version = "V2"
    training_data_version = "v2"
    mode = "optimized"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_to_optimized = "gridsearch_v3_correct" #f"./sampled_data_mode_{mode}_version_{version}_N-{N}_K-{K}.csv"
    path_to_random = "./gridsearch/GS_random.csv"

    '''
    # Get simulated data with random pp sampled from grid
    S = Sampler(lower_lims=lower_lims, upper_lims=upper_lims, N=N, T=T, version=version, device=device,path_to_optimized=path_to_optimized)
    batch,pandemic_parameters,starting_points = S(K=K,L=L,B=B,mode=mode) 
    S.__plot_subset____(L=L, K=K, B=B, T=T, starting_points=starting_points, batch=batch, path="./gridsearch/samplertest.png")
    S.__save_pp_cost_to_file__(N=N, K=K, B=B, L=L, T=T, version=version, mode=mode, batch=batch,pp=pandemic_parameters,path=path_to_random)
    '''

    # Get simulated data with optimized pp from gridsearch 
    S = Sampler(lower_lims=lower_lims, upper_lims=upper_lims, N=N, T=T, version=version, device=device,path_to_optimized=path_to_optimized)
    batch,pandemic_parameters,starting_points = S(K=K,L=L,B=B,mode=mode,mse_threshold=mse_threshold) 
    S.__plot_simulation_subset____(L=L, K=K, B=B, T=T, starting_points=starting_points, batch=batch, path=f"./trainingdata/{training_data_version}/simulation_samples_{training_data_version}.png")
    S.__save_to_file_for_training__(data=batch, pandemic_parameters=pandemic_parameters, version_name=training_data_version)
    data, pp = S.__load_from_training_file__(version_name=training_data_version)
    print(data.shape)
    print(pp.shape)

'''  
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
'''