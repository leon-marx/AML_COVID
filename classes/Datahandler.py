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