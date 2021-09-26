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

    reps = 10
    num_pp_eval = 50

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
            optim_pp_list = df_pp['0'][:num_pp_eval]

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
            plt.boxplot(all_pps,labels=keys)
            plt.title(f'{country} | wave {wave_id} | N = {N[0]} | Best {num_pp_eval} pp combinations')
            plt.savefig(f"./data/boxplots/boxplot_{country}_wave_{wave_id}_paper_average_{num_pp_eval}_initial.jpg")