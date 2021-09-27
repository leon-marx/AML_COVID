from sys import version
from numpy.core.fromnumeric import mean
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
import random
import os
import tqdm


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
            W = World(N = params["N"], D = params["D"], r = params["r"], d = params["d"], N_init= params["N_init"],epsilon=params["epsilon"],version = params["version"], device=self.device)
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
            self.cumulative -= self.cumulative[0].item()
            self.cumulative /= self.cumulative[-1].item()

        elif mode == "SIR":
            Timeseries = create_toydata(T = params["T"], I0 = params["I0"], R0 = params["R0"], N = params["N"], beta = params["beta"], gamma = params["gamma"]).T
            self.cumulative = torch.tensor((Timeseries[1]+Timeseries[2]) / params["N"]).to(device)

            #Normalize with the final value of the real data
            


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

    #plt.savefig(f"./compare_hard_smooth_transition.jpg")

    plt.show()

def compare_real_data_simulation(reps = 2,n = 1):

    #params_simulation = {'N': 1500, 'version': 'V2', 'd': 5.333, 'D': 6, 'r': 0.05, 'r_new': 0.05, 'D_new': 6, 'T_change_D': 91, 'Smooth_transition': 1, 'N_init': 9, 'T': 90, 'epsilon': 0.04}
    #params_simulation = {'N': 1500, 'version': 'V2', 'd': 5.333, 'D': 5, 'r': 0.045, 'r_new': 0.05, 'D_new': 6, 'T_change_D': 91, 'Smooth_transition': 1, 'N_init': 11, 'T': 90, 'epsilon': 0.05}
    params_simulation = {'N': 1500, 'version': 'V2', 'd': 5.333, 'D': 5, 'r': 0.045, 'r_new': 0.05, 'D_new': 6, 'T_change_D': 91, 'Smooth_transition': 1, 'N_init': 11, 'T': 90, 'epsilon': 0.05}
    params_real = {"file":"Israel.txt","wave":3,"full":False,"use_running_average":True,"dt_running_average":14}

    fs = 30

    params_simulation["Smooth_transition"] = 1
    DH = DataHandler("Real",params_real,device = "cpu")
    real_data,starting_points  = DH(B = None,L = None,return_plain=True)

    params_simulation["T"] = len(real_data)

    set = np.zeros([reps,params_simulation["T"]])


    for i in tqdm.tqdm(range(reps)):
        DH = DataHandler("Simulation",params_simulation,device = "cpu")
        batch2,starting_points  = DH(B = None,L = None,return_plain=True)
        set[i] = batch2

    #Get the mean 
    means = np.mean(set,axis=0)
    std = np.std(set,axis=0)


    plt.figure(figsize = (30,15))

    means -= means[0]

    plt.fill_between(x = np.arange(len(means)),y1=means - std,y2 = means + std,color = "orange")
    plt.plot(real_data,color = "b",label = "Real",linewidth = 4)
    plt.plot(means,ls = ":",color = "r",label = "mean simulation",linewidth = 4)

    plt.plot(means + std,color = "r")
    plt.plot(means - std,color = "r",label = r"1 $\sigma$-interval")

    
    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases [days]",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.legend(fontsize = fs)

    plt.savefig(f"./compare_simulation_real_data_{params_real['file'].split('.')[0]}_{params_real['wave']}_{reps}_reps_{n}.jpg")
    plt.show()

def compare_SIR_Simulation():
    ############################################################################
    #Case D = 8
    ############################################################################
    params_simulation = {'N': 1000, 'version': 'V2', 'd': 6, 'D': 8, 'r': 0.1, 'r_new': 0.05, 'D_new': 6, 'T_change_D': 1e6, 'Smooth_transition': 1, 'N_init': 10, 'T': 100, 'epsilon': 0.1}
    params_SIR = {'N': 1000, 'T':params_simulation['T'], 'I0':params_simulation['N_init'], 'R0':params_simulation['N_init'], 'beta':params_simulation['r'] * params_simulation['D'] / params_simulation['N'],'gamma':1 / params_simulation['d']}

    reps = 25

    fs = 30

    sum_sim_V2 = np.zeros(params_simulation['T'])
    sum_sim_V3 = np.zeros(params_simulation['T'])
    sum_SIR = np.zeros(params_simulation['T'])

    for i in tqdm.tqdm( range(reps)):
        DH_simulation_V2 = DataHandler("Simulation",params_simulation,device = "cpu")
        ts_sim_V2,_ = DH_simulation_V2(None,None,return_plain=True)
        sum_sim_V2 += ts_sim_V2.numpy()

        DH_simulation_V3 = DataHandler("Simulation",params_simulation,device = "cpu")
        ts_sim_V3,_ = DH_simulation_V3(None,None,return_plain=True)
        sum_sim_V3 += ts_sim_V3.numpy()

        DH_SIR = DataHandler("SIR",params_SIR,device = "cpu")
        ts_SIR,_ = DH_SIR(None,None,return_plain=True)
        sum_SIR += ts_SIR.numpy()

    sum_sim_V2 /= reps
    sum_sim_V3 /= reps
    sum_SIR /= reps

    plt.figure(figsize = (45,10))

    plt.subplot(1,2,1)
    plt.title("D = 8",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases []",fontsize = fs)

    plt.plot(sum_sim_V2,color = "b",label = "simulation, fixed degree",linewidth = 4)
    plt.plot(sum_sim_V3,color = "r",label = "simulation, Poisson distributed degree",linewidth = 4)
    plt.plot(sum_SIR,color = "y",label = "SIR",linewidth = 4)

    plt.legend(fontsize = fs,loc = 4)

    ############################################################################
    #Case D = 3
    ############################################################################
    params_simulation = {'N': 1000, 'version': 'V2', 'd': 6, 'D': 3, 'r': 0.1, 'r_new': 0.05, 'D_new': 6, 'T_change_D': 1e6, 'Smooth_transition': 1, 'N_init': 10, 'T': 100, 'epsilon': 0.1}
    params_SIR = {'N': 1000, 'T':params_simulation['T'], 'I0':params_simulation['N_init'], 'R0':params_simulation['N_init'], 'beta':params_simulation['r'] * params_simulation['D'] / params_simulation['N'],'gamma':1 / params_simulation['d']}

    sum_sim_V2 = np.zeros(params_simulation['T'])
    sum_sim_V3 = np.zeros(params_simulation['T'])
    sum_SIR = np.zeros(params_simulation['T'])

    for i in tqdm.tqdm( range(reps)):
        DH_simulation_V2 = DataHandler("Simulation",params_simulation,device = "cpu")
        ts_sim_V2,_ = DH_simulation_V2(None,None,return_plain=True)
        sum_sim_V2 += ts_sim_V2.numpy()

        DH_simulation_V3 = DataHandler("Simulation",params_simulation,device = "cpu")
        ts_sim_V3,_ = DH_simulation_V3(None,None,return_plain=True)
        sum_sim_V3 += ts_sim_V3.numpy()

        DH_SIR = DataHandler("SIR",params_SIR,device = "cpu")
        ts_SIR,_ = DH_SIR(None,None,return_plain=True)
        sum_SIR += ts_SIR.numpy()

    sum_sim_V2 /= reps
    sum_sim_V3 /= reps
    sum_SIR /= reps

    plt.subplot(1,2,2)
    plt.title("D = 3",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases []",fontsize = fs)

    plt.plot(sum_sim_V2,color = "b",label = "simulation, fixed degree",linewidth = 4)
    plt.plot(sum_sim_V3,color = "r",label = "simulation, Poisson distributed degree",linewidth = 4)
    plt.plot(sum_SIR,color = "y",label = "SIR",linewidth = 4)

    plt.legend(fontsize = fs,loc = 4)
    plt.savefig(f"compare_simulation_SIR_reps_{reps}.jpg")
