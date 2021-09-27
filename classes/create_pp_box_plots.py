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


if __name__ == "__main__":

    #num_pp_eval = 50
    fs = 30

    cost_threshold = 0.01

    with open("./classes/Countries/wave_regions.json","r") as file:
        waves = json.load(file)
    countries = waves.keys()
    
    for country in countries:
        if country == 'Israel' or country == 'UnitedKingdom':
            continue
        waves_list = waves[country]
        n_waves = waves_list['N_waves']

        for wave_id in range(1,n_waves+1):
            
            # get optimal pps 
            df_pp = pd.read_csv(f'./data/gridsearch_results/GS_fit_{country}_{wave_id}.csv')      
            costs = df_pp['1']
            optim_pp_list = df_pp['0'][costs<=0.005] #[:num_pp_eval]
            num_pp_eval = len(optim_pp_list)

            # delete version, smooth-transition and N 
            optim_pp_list_reduced, N, T, T_change_D = list(), list(), list(), list()
            for i in range(len(optim_pp_list)):
                tmp = ast.literal_eval(optim_pp_list[i])
                tmp.pop('version')
                tmp.pop('Smooth_transition')
                tmp.pop('D_new')
                tmp.pop('r_new')
                N.append(tmp.pop('N'))
                T.append(tmp.pop('T'))
                T_change_D.append(tmp.pop('T_change_D'))
                optim_pp_list_reduced.append(tmp)
                
            keys = list(tmp.keys())
            all_pps = np.zeros((num_pp_eval,len(keys)))

            for i in range(len(optim_pp_list_reduced)):
                all_pps[i] = np.array(list(optim_pp_list_reduced[i].values()))

            plt.figure(figsize = (30,15))
            boxprops = dict(linewidth=2)
            whiskerprops=dict(linewidth=1.0)
            plt.boxplot(all_pps,labels=keys, boxprops=boxprops, whiskerprops=whiskerprops)
            plt.xlabel("pandemic parameters", fontsize=fs)
            plt.xticks(fontsize=fs*3/4, rotation=90)
            plt.yticks(fontsize=fs*3/4, rotation=90)
            plt.title(f'{country} | wave {wave_id} | N = {N[0]} | Best {num_pp_eval} pp combinations with loss <= {cost_threshold}',fontsize=fs)
            plt.savefig(f"./data/boxplots/boxplot_{country}_wave_{wave_id}_paper_average_{num_pp_eval}_initial_new.jpg")