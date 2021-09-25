from matplotlib import pyplot as plt
from numpy.lib.financial import ppmt
from Datahandler import DataHandler
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import random
import copy
from operator import itemgetter
import tqdm
import pandas as pd
import json

#load the Time series for the united states
data_usa = np.loadtxt("./Countries/UnitedStates.txt",skiprows=4)
data_usa = data_usa[57:107] 
data_usa -= data_usa[0]

#select the initial cases
initial = data_usa[(data_usa <= 1e-3)]
initial = torch.tensor(initial)

pp_grid_reduced = {
    "epsilon": [0.05,0.04,0.03],
    "D": [8,9,10],
    "D_new": [8],
    "r": [0.05,0.04,0.03],
    "r_new": [8],
    "d": [4,5,6],
    "N_init": [1,2,3],
    "T_change_D":[1e4],
    "Smooth_transition":[1]
}

class GridSearch_PP_finder():
    def __init__(self, pp_grid, eval_num = 1, N_pop = 5000, version = "V2",device = "cuda" if torch.cuda.is_available() else "cpu", mode="full"):
        '''
        parameters:
            pp_grid                 grid for pp search space with epsilon D,r,d,N_init
            params_real             Paramters describing the real time series
            N_pop                   Number of individuals in the small world model
            version                 Version of the Small world model                
            device                  Device
            mode                    if "random" -> sample pps in intervals specified in pp_grid
                                    if "full_grid" -> use cartesian product of pp specified in pp_grid
        '''
        assert all(key in pp_grid for key in ['epsilon','d','D','r','N_init'])
        self.mode = mode
        self.eval_num = eval_num
        self.pp_grid = pp_grid
        self.version = version
        self.device = device
        self.simulation_parameters = {
            "N":N_pop,
            "version": version
        }

    def cost(self,ts_1,ts_2):
        '''
        returns the mean squared error between the two time series

        parameters:
            ts_1    first time series
            ts_2    second time series

        returns:
            mse
        '''

        mse = (ts_1 - ts_2)**2

        return torch.mean(torch.tensor(mse))

    def get_time_series(self,mode,parameters):
        '''
        Get the time series used in the optimization and shape it as required

        parameters:
            mode:           Real or simulated data
            parameters:     parameters used to generate the time series
            device:         Device

        returns:
            ts:             Generated time serie
        '''
        if mode == "Real":
            return initial.numpy()

        else:
            DH = DataHandler(mode,parameters,device = "cpu")
            ts,starting_points  = DH(B = None,L = None,return_plain=True)

            #delete offset
            ts = ts - ts[0]

            #Normalize
            ts = ts / ts[-1]

            #Get correct final value

            ts *= initial[-1]

            #plt.plot(ts)
            #plt.show()

            return ts

    def __call__(self, max_evals=1000):
        #Get the real time series
        ts_real = self.get_time_series("Real", None)
 
        #Get the number of time steps
        T = len(ts_real)

        # try all pp combinations
        keys, values = zip(*self.pp_grid.items())
        pp_combinations = list(itertools.product(*values))

        if(self.mode == "random"):
            pp_combinations = random.sample(pp_combinations,self.eval_num)

        i = 0
        results = list()
        print(f"Number of evaluated combinations: {len(pp_combinations)}")

        for i in tqdm.tqdm(range(len(pp_combinations))):#v in pp_combinations:

            v = pp_combinations[i]
            # Parse pp
            pp = dict(zip(keys, v))
            self.simulation_parameters['d'] = pp['d']
            self.simulation_parameters['D'] = pp['D']
            self.simulation_parameters['r'] = pp['r']
            self.simulation_parameters['r_new'] = pp["r_new"]
            self.simulation_parameters['D_new'] = pp["D_new"]
            self.simulation_parameters['T_change_D'] = pp["T_change_D"]
            self.simulation_parameters['Smooth_transition'] = pp["Smooth_transition"]
            self.simulation_parameters['N_init'] = pp['N_init']
            self.simulation_parameters['T'] = T
            self.simulation_parameters['epsilon'] = pp['epsilon']

            # Without changing D 
            reps = 15
            self.simulation_parameters["T_change_D"] = T  
            set = np.zeros([reps,self.simulation_parameters['T']])
            

            for i in range(reps):
                ts = self.get_time_series("Simulation", self.simulation_parameters)

                ts = ts.numpy()

                ts -= ts[0]
                ts /= ts[len(initial)-1]
                ts *= initial.numpy()[-1]

                set[i] = ts

            means = np.mean(set,axis=0)

            #Get the mse between the simulation and the real time series
            mse = self.cost(ts_real,means)
            #print(f"PP: {pp} | MSE = {mse}")

            #Append 
            current_simulation_parameters = copy.deepcopy(self.simulation_parameters)
            results.append([current_simulation_parameters, float(mse.cpu().numpy())])

        # Sort by mse-loss
        results.sort(key=itemgetter(1), reverse=False)

        #Get the best parameters
        mse_list = [results[i][1] for i in range(len(results))] 
        index = np.argmin(mse_list)
        optimal_pp = results[index][0]

        # save results to disk
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"./data_find_best_pp_USA_paper.csv") 


        #Get the time series for the best parameters
        self.simulation_parameters["D"] = int(optimal_pp['D'])
        self.simulation_parameters["D_new"] = int(optimal_pp["D_new"])
        self.simulation_parameters["r"] = optimal_pp['r']
        self.simulation_parameters["r_new"] = optimal_pp["r_new"]

        self.simulation_parameters['T_change_D'] = int(pp["T_change_D"])
        self.simulation_parameters['Smooth_transition'] = int(pp["Smooth_transition"])

        self.simulation_parameters["N_init"] = int(optimal_pp['N_init'])
        self.simulation_parameters["epsilon"] = optimal_pp['epsilon']
        ts_optimal = self.get_time_series("Simulation",self.simulation_parameters)
        
        #plot the best fitting simulation with observed data 
        plt.figure(figsize=(20,10))
        plt.plot(ts_optimal,label = "simulation")
        plt.plot(ts_real,label = "observation")
        plt.legend()
        plt.savefig(f"./USA_paper.jpg")
        plt.close()

        return optimal_pp


print("Get the optimal parameters for the initial sequence...")
#finder = GridSearch_PP_finder(pp_grid_reduced)
#optimal_pp = finder()
#print("\tOptimal pp: ",optimal_pp)

print("Compare the full time series...")

fs = 30
reps = 500
#optimal_pp["T"] = len(data_usa)

optimal_pp =  {'N': 5000, 'version': 'V2', 'd': 4, 'D': 8, 'r': 0.05, 'r_new': 8, 'D_new': 8, 'T_change_D': 1000, 'Smooth_transition': 1, 'N_init': 1, 'T': 17, 'epsilon': 0.05}
optimal_pp["T"] = len(data_usa)

set = np.zeros([reps,optimal_pp["T"]])

for i in tqdm.tqdm(range(reps)):
    DH = DataHandler("Simulation",optimal_pp,device = "cpu")
    ts,starting_points  = DH(B = None,L = None,return_plain=True)

    ts = ts.numpy()

    ts -= ts[0]

    
    if ts[len(initial)-1] == 0: continue

    ts /= ts[len(initial)-1]
    ts *= initial.numpy()[-1]

    set[i] = ts

#Get the mean 
set = set[(set[:,-1])!= 0]
means = np.mean(set,axis=0)
std = np.std(set,axis=0)

plt.figure(figsize = (30,15))

means -= means[0]

plt.fill_between(x = np.arange(len(means)),y1=means - std,y2 = means + std,color = "orange")
plt.plot(data_usa,color = "b",label = "Real",linewidth = 4)
plt.plot(means,ls = ":",color = "r",label = "mean simulation",linewidth = 4)

plt.plot(means + std,color = "r")
plt.plot(means - std,color = "r",label = r"1 $\sigma$-interval")


plt.xlabel("time [days]",fontsize = fs)
plt.ylabel("cumulative cases [days]",fontsize = fs)
plt.xticks(fontsize = fs)
plt.yticks(fontsize = fs)
plt.legend(fontsize = fs)

plt.savefig(f"./USA_paper_average_{reps}.jpg")
plt.show()

with open("optimal_pp.json","w") as file:
    json.dump(optimal_pp,file)

