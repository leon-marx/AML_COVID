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
import os 

with open("./classes/Countries/wave_regions.json","r") as file:
    waves = json.load(file)
countries = waves.keys()

for country in countries:
    waves_list = waves[country]
    n_waves = waves_list['N_waves']

    #load the Time series for the united states
    cum_inf_data_country = np.loadtxt(f"./classes/Countries/{country}.txt",skiprows=4)
    # cum_inf_data_country = cum_inf_data_country[waves[country][wave_id][0]:waves[country][wave_id][1]]

    fs = 30
    lw=2

    plt.figure(figsize = (30,15))
    plt.title(f"{country}",fontsize=fs)
    plt.plot(cum_inf_data_country,color = "b",label = "Real",linewidth = lw)
    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases / population size",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    for wave_id in range(1,n_waves+1):
        wave_id = str(wave_id)
        limits = waves_list[wave_id]
        plt.axvspan(waves[country][wave_id][0], waves[country][wave_id][1], color='goldenrod', alpha=0.2)
        plt.axvline(waves[country][wave_id][0], color='goldenrod', linewidth=2.0, alpha=0.9)
        plt.axvline(waves[country][wave_id][1], color='goldenrod', linewidth=2.0, alpha=0.9)
    plt.legend(fontsize = fs)

    plt.savefig(f"./data/real_cumulative_curves/{country}_with_waves_corrected.png")


'''
files = os.listdir("./classes/Countries")

for file in files:
    if file.endswith('.txt'):
        data_country = np.loadtxt("./classes/Countries/" + file,skiprows=4)
        fs = 30
        plt.figure(figsize = (30,15))
        plt.title(str(file.replace('.txt','')))
        plt.plot(data_country,color = "b",label = "Real",linewidth = 4)
        plt.xlabel("time [days]",fontsize = fs)
        plt.ylabel("cumulative cases [days]",fontsize = fs)
        plt.xticks(fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.axvspan(low_bnd, up_bnd, color='red', alpha=0.5)
        plt.legend(fontsize = fs)

        plt.savefig(f"./real_cumulative_curves/{file.replace('.txt','')}.jpg")
'''