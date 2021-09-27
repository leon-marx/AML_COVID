import os
from matplotlib import pyplot as plt
#from numpy.lib.financial import ppmt
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
import ast

class GridSearch_PP_finder():
    def __init__(self, pp_grid, eval_num = 10, N_pop = 1000, version = "V2",device = "cuda" if torch.cuda.is_available() else "cpu", mode="full"):
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
        self.reps = 10

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
            reps = self.reps
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


if __name__ == "__main__":

    #print("Get the optimal parameters for the initial sequence...")
    #finder = GridSearch_PP_finder(pp_grid_reduced)
    #optimal_pp = finder()
    #print("\tOptimal pp: ",optimal_pp)

    #print("Compare the full time series...")
    fs = 30
    reps = 5
    num_pp_eval = 1

    #with open("./classes/Countries/wave_regions.json","r") as file:
    #    waves = json.load(file)
    #countries = waves.keys()

    countries = ['UnitedStates']
    waves = {'UnitedStates': {
        'N_waves': 1,
        "1":[40,120]
        }}
    '''
    countries = ['Germany']
    waves = {'Germany': {
        'N_waves': 2,
        "2":[250,400]
        }}
    '''

    for country in countries:
        waves_list = waves[country]
        n_waves = waves_list['N_waves']

        for wave_id in range(1,n_waves+1):
            if country == "Germany" and wave_id == 1:
                continue 
            print(f'{country}: {wave_id}')
            wave_id = str(wave_id)
            limits = waves_list[wave_id]

            #load the Time series for the united states
            cum_inf_data_country = np.loadtxt(f"./classes/Countries/{country}.txt",skiprows=4)
            cum_inf_data_country = cum_inf_data_country[waves[country][wave_id][0]:waves[country][wave_id][1]]
            
            #select the initial cases
            #initial = cum_inf_data_country[(cum_inf_data_country <= 1e-3)]
            #initial = torch.tensor(initial)

            #optimal_pp = {'N': 2500, 'version': 'V2', 'd': 5.833333333333333, 'D': 3, 'r': 0.1, 'r_new': 0.06, 'D_new': 9, 'T_change_D': 136, 'Smooth_transition': 10000.0, 'N_init': 8, 'T': 136, 'epsilon': 0.01}
            df_pp_gs = pd.read_csv(f'./data/gridsearch_results/GS_fit_{country}_{wave_id}.csv')      
            optimal_pp_list_gs = df_pp_gs['0'][:num_pp_eval]

            optimal_pp_list_gp_usa_1 = [{
                "T_change_D": 76,
                "T": 76,
                "Smooth_transition": 10000,
                "r_new": 0,
                "D_new": 8, 
                "epsilon": 0.01, 
                "N_init": 3, 
                "r": 0.07,
                "D": 9,
                "N":2500,
                "d":6,
                "version": "V2"
            }]

            optimal_pp_list_gp_germany_2 = [{
                "T_change_D": 136,
                "T": 136,
                "Smooth_transition": 10000,
                "r_new": 0,
                "D_new": 1, 
                "epsilon": 0.01, 
                "N_init": 3, 
                "r": 0.07,
                "D": 7,
                "N":2500,
                "d":6,
                "version": "V2"
            }]

            optimal_pp_list_gp = optimal_pp_list_gp_usa_1 #optimal_pp_list_gp_germany_2#

            # count number of pps per run 
            rank = 1

            for optimal_pp in optimal_pp_list_gs:
                
                optimal_pp = ast.literal_eval(optimal_pp)

                #select the initial cases
                #initial = data_country[(data_country <= 1e-3)]
                #initial = torch.tensor(initial)
                #optimal_pp["T"] = len(initial)

                set_gs = np.zeros([reps,optimal_pp["T"]])

                ###############################################################################################################################################################################################
                #mean for th efirst small part
                for i in tqdm.tqdm(range(reps)):
                    DH = DataHandler("Simulation",optimal_pp,device = "cpu")
                    ts_gs,starting_points  = DH(B = None,L = None,return_plain=True)

                    ts_gs = ts_gs.numpy()

                    print(len(ts_gs))

                    #ts -= ts[0]
                    #ts /= ts[len(ts)-1]
                    #ts *= cum_inf_data_country[len(ts)-1]
                    
                    #cum_inf_data_country -= cum_inf_data_country[0]
                    #cum_inf_data_country /= cum_inf_data_country[-1]

                    #if len(initial) == 0:
                    #    if ts[0] == 0:
                    #        ts *= cum_inf_data_country[0]
                    #    else:
                    #        ts /= ts[0] # substract offest
                    #        ts *= cum_inf_data_country[0] # divide by last value
                    
                    #else:                       
                    #
                    #    if ts[len(initial)-1] == 0: continue

                    #    ts /= ts[len(initial)-1]
                    #       ts *= initial.numpy()[-1]

                    set_gs[i] = ts_gs

                #Get the mean 
                set_gs = set_gs[(set_gs[:,-1])!= 0]
                means_gs = np.mean(set_gs,axis=0)
                std_gs = np.std(set_gs,axis=0)
                means_gs -= means_gs[0]

                set_gp = np.zeros([reps,optimal_pp["T"]])
                for i in tqdm.tqdm(range(reps)):
                    DH = DataHandler("Simulation",optimal_pp,device = "cpu")
                    ts_gp,starting_points  = DH(B = None,L = None,return_plain=True)

                    ts_gp = ts_gp.numpy()

                    print(len(ts_gp))

                    #ts -= ts[0]
                    #ts /= ts[len(ts)-1]
                    #ts *= cum_inf_data_country[len(ts)-1]

                    #if len(initial) == 0:
                    #    if ts[0] == 0:
                    #        ts *= cum_inf_data_country[0]
                    #    else:
                    #        ts /= ts[0] # substract offest
                    #        ts *= cum_inf_data_country[0] # divide by last value
                    
                    #else:                       
                    #
                    #    if ts[len(initial)-1] == 0: continue

                    #    ts /= ts[len(initial)-1]
                    #       ts *= initial.numpy()[-1]

                    set_gp[i] = ts_gp

                #Get the mean 
                set_gp = set_gp[(set_gp[:,-1])!= 0]
                means_gp = np.mean(set_gp,axis=0)
                std_gp = np.std(set_gp,axis=0)
                means_gp -= means_gp[0]

                plt.figure(figsize = (30,15))

                plt.suptitle(f"{country} | {wave_id}", fontsize=fs)

                #normalize length
                cum_inf_data_country = cum_inf_data_country[0:len(means_gp)]
                cum_inf_data_country -= cum_inf_data_country[0]
                cum_inf_data_country /= cum_inf_data_country[len(means_gp)-1]

                plt.fill_between(x = np.arange(len(means_gp)),y1=means_gp - std_gp,y2 = means_gp + std_gp,color = "r", alpha=0.3)
                plt.fill_between(x = np.arange(len(means_gp)),y1=means_gs - std_gs,y2 = means_gs + std_gs,color = "g", alpha=0.3)
                
                plt.plot(cum_inf_data_country[:len(means_gp-1)],color = "b",label = "real",linewidth = 4)

                plt.plot(means_gp,ls = ":",color = "r",label = "mean simulation gp",linewidth = 4)
                plt.plot(means_gp + std_gp,color = "r")
                plt.plot(means_gp - std_gp,color = "r",label = r"1 $\sigma$-interval")

                plt.plot(means_gs,ls = ":",color = "g",label = "mean simulation gs",linewidth = 4)
                plt.plot(means_gs + std_gs,color = "g")
                plt.plot(means_gs - std_gs,color = "g",label = r"1 $\sigma$-interval")


                plt.xlabel("time [days]",fontsize = fs)
                plt.ylabel("cumulative cases",fontsize = fs)
                plt.xticks(fontsize = fs)
                plt.yticks(fontsize = fs)
                plt.legend(fontsize = fs)

                plt.savefig(f"./{country}_wave_{wave_id}_rank_{rank}_paper_average_{reps}_full_GP_directcompare.jpg")
                
                rank+=1