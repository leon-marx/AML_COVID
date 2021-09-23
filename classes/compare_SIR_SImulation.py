from Datahandler import DataHandler
import matplotlib.pyplot as plt
import numpy as np
def plotter(time_Series,labels):
    fs = 30

    for i in range(len(time_Series)):
        plt.plot(time_Series[i],label = labels[i])
        

    
    plt.legend(fontsize = fs)

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("cumulative cases []",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.ylim([0,1.2])



#PArameters for the different modes
params_simulation_V2 = {
    "D": 8,
    "N": 1000,
    "r": 0.1,
    "d": 6,
    "N_init": 10,
    "T":100,
    "epsilon":0.1,
    "version":"V2"
}


Sum_V2_im_1 = np.zeros(params_simulation_V2["T"])
Sum_V2_im_2 = np.zeros(params_simulation_V2["T"])

Sum_V3_im_1 = np.zeros(params_simulation_V2["T"])
Sum_V3_im_2 = np.zeros(params_simulation_V2["T"])

Sum_SIR_im1 = np.zeros(params_simulation_V2["T"])
Sum_SIR_im2 = np.zeros(params_simulation_V2["T"])

reps = 1000
for i in range(reps):
    print(f"i = {i}")
    #############################################################
    #Same Results for Simulation and for SIR
    #############################################################
    params_simulation_V2["D"] = 8

    params_simulation_V3 = {
        "D": params_simulation_V2["D"],
        "N": params_simulation_V2["N"],
        "r": params_simulation_V2["r"],
        "d": params_simulation_V2["d"],
        "N_init": params_simulation_V2["N_init"],
        "T":params_simulation_V2["T"],
        "epsilon":params_simulation_V2["epsilon"],
        "version":"V3"
    }

    #Translate the parameters of the simulation into the SIR model parameters
    gamma = 1 / params_simulation_V2["d"]
    beta = params_simulation_V2["r"] * params_simulation_V2["D"] / params_simulation_V2["N"]

    params_SIR = {
        "T":params_simulation_V2["T"],
        "I0":params_simulation_V2["N_init"],
        "R0":0,
        "N":params_simulation_V2["N"],
        "beta":beta,
        "gamma":gamma
    }

    #initialize the data handler
    DH_Sim_V2 = DataHandler("Simulation",params_simulation_V2,device = "cpu")
    DH_Sim_V3 = DataHandler("Simulation",params_simulation_V3,device = "cpu")
    DH_SIR = DataHandler("SIR",params_SIR,device = "cpu")

    #Get the time series
    ts_V2,_ = DH_Sim_V2(None,None,True)
    ts_V3,_ = DH_Sim_V3(None,None,True)
    ts_SIR,_ = DH_SIR(None,None,True)

    Sum_V2_im_1 += ts_V2
    Sum_V3_im_1 += ts_V3
    Sum_SIR_im1 += ts_SIR

    #############################################################
    #Different Results for Simulation and for SIR
    #############################################################
    params_simulation_V2["D"] = 3
    params_simulation_V3["D"] = params_simulation_V2["D"]
    params_SIR["beta"] = params_simulation_V2["r"] * params_simulation_V2["D"] / params_simulation_V2["N"]

    DH_Sim_V2 = DataHandler("Simulation",params_simulation_V2,device = "cpu")
    DH_Sim_V3 = DataHandler("Simulation",params_simulation_V3,device = "cpu")
    DH_SIR = DataHandler("SIR",params_SIR,device = "cpu")

    #Get the time series
    ts_V2,_ = DH_Sim_V2(None,None,True)
    ts_V3,_ = DH_Sim_V3(None,None,True)
    ts_SIR,_ = DH_SIR(None,None,True)

    Sum_V2_im_2 += ts_V2
    Sum_V3_im_2 += ts_V3
    Sum_SIR_im2 += ts_SIR

Sum_V2_im_1 /= reps
Sum_V3_im_1 /= reps
Sum_SIR_im1 /= reps

Sum_V2_im_2 /= reps
Sum_V3_im_2 /= reps
Sum_SIR_im2 /= reps

plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
plt.title("D = 8",fontsize = 30)
plotter([Sum_SIR_im1,Sum_V2_im_1,Sum_V3_im_1],["SIR","Simulation V2","Simulation V3"])

plt.subplot(1,2,2)
plt.title("D = 3",fontsize = 30)
plotter([Sum_SIR_im2,Sum_V2_im_2,Sum_V3_im_2],["SIR","Simulation V2","Simulation V3"])
plt.savefig(f"compare_SIR_simulation_as_in_Paper_average_{reps}_reps.jpg")