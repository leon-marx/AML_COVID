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
        plt.legend(fontsize = fs)

        plt.savefig(f"./real_cumulative_curves/{file.replace('.txt','')}.jpg")