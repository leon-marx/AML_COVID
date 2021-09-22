from Datahandler import DataHandler
import matplotlib.pyplot as plt

def plotter(time_Series,labels):
    fs = 30

    plt.figure(figsize = (20,10))

    for i in range(len(time_Series)):
        plt.plot(time_Series[i],label = labels[i])
        

    
    plt.legend(fontsize = fs)

    plt.xlabel("time [days]",fontsize = fs)
    plt.ylabel("time [days]",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)

    plt.savefig("compare-sIR-simulation.jpg")
    plt.show()

#PArameters for the different modes
params_simulation_V2 = {
    "D": 8,
    "N": 1000,
    "r": 0.1,
    "d": 6,
    "N_init": 10,
    "T":50,
    "epsilon":0.1,
    "version":"V2"
}

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


#plot the results
plotter([ts_SIR,ts_V2,ts_V3],["SIR","Simulation V2","Simulation V3"])
